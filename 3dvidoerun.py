import open3d as o3d
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
from tqdm import tqdm

class TRTLoader:
    """
    Loads a TensorRT engine and handles the inference process.
    (This class is from your code, unchanged)
    """
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()

        for binding in self.engine:
            shape = self.engine.get_tensor_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        self.input_shape = self.engine.get_tensor_shape(self.engine[0])

    def infer(self, image):
        np.copyto(self.inputs[0]['host'], image.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_v2(bindings=self.bindings)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host']

class ReconstructionProcessor:
    """
    Manages the entire 3D reconstruction process from a video stream.
    """
    def __init__(self, engine_path, intrinsics_matrix, voxel_size=0.01):
        # --- Core Components ---
        print("Loading TensorRT engine...")
        self.trt_model = TRTLoader(engine_path)
        _, _, _, self.h, self.w = self.trt_model.input_shape
        self.intrinsics_matrix = intrinsics_matrix
        self.intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
            self.w, self.h, intrinsics_matrix[0, 0], intrinsics_matrix[1, 1],
            intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]
        )

        # --- SIFT for Feature Matching ---
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # --- Reconstruction State ---
        self.global_pose = np.eye(4)
        self.prev_color_frame = None
        self.prev_depth_map = None

        # --- TSDF Volume for Clean Fusion ---
        print("Initializing TSDF Volume...")
        self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=0.04, # Voxel size * 2 is a good rule of thumb
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        self.is_initialized = False

    def get_depth_from_model(self, color_frame_bgr):
        """Preprocesses a frame and runs inference to get its depth map."""
        input_image = cv2.resize(color_frame_bgr, (self.w, self.h))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image_norm = input_image.astype(np.float32) / 255.0
        input_image_norm = input_image_norm.transpose(2, 0, 1)
        input_image_norm = np.expand_dims(input_image_norm, axis=0)
        raw_depth = self.trt_model.infer(input_image_norm)
        depth_map = raw_depth.reshape(self.h, self.w)
        return depth_map

    def estimate_pose_and_scale(self, color1, color2, depth1):
        """Calculates the relative transformation and scale between two frames."""
        kp1, des1 = self.sift.detectAndCompute(color1, None)
        kp2, des2 = self.sift.detectAndCompute(color2, None)

        matches = self.flann.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) < 20:
            return None, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        E, mask = cv2.findEssentialMat(pts2, pts1, self.intrinsics_matrix, method=cv2.RANSAC, prob=0.99, threshold=1.0)
        if E is None: return None, None
        
        _, R, t, mask = cv2.recoverPose(E, pts2, pts1, self.intrinsics_matrix, mask=mask)

        # --- Solve for scale ---
        scales = []
        for i in range(len(good_matches)):
            if mask[i, 0]:
                pt1_3d_z = depth1[int(pts1[i, 1]), int(pts1[i, 0])]
                if pt1_3d_z > 0:
                    pt1_3d = np.linalg.inv(self.intrinsics_matrix) @ (np.array([pts1[i, 0], pts1[i, 1], 1.0]) * pt1_3d_z)
                    pt1_3d_transformed = R @ pt1_3d + t.flatten()
                    expected_depth_of_pt2 = pt1_3d_transformed[2]
                    
                    # Using depth1 for pt2 as a proxy since the true scale of depth2 is unknown.
                    pt2_3d_z = depth1[int(pts2[i, 1]), int(pts2[i, 0])]
                    if expected_depth_of_pt2 > 0 and pt2_3d_z > 0:
                        ratio = pt2_3d_z / expected_depth_of_pt2
                        if 0.5 < ratio < 2.0:
                            scales.append(ratio)
        
        if not scales: return None, None
        
        scale = np.median(scales)
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = t.flatten() * scale
        
        return transformation, scale

    def process_frame(self, color_frame_bgr):
        """Processes a single video frame and integrates it into the 3D model."""
        new_depth_map = self.get_depth_from_model(color_frame_bgr)
        resized_color_bgr = cv2.resize(color_frame_bgr, (self.w, self.h))

        if not self.is_initialized:
            # First frame: initialize the reconstruction at the origin
            self.prev_color_frame = resized_color_bgr
            self.prev_depth_map = new_depth_map
            
            # Use a fixed scale for the very first depth map
            depth_scaled = new_depth_map * 10.0 

            color_o3d = o3d.geometry.Image(cv2.cvtColor(resized_color_bgr, cv2.COLOR_BGR2RGB))
            depth_o3d = o3d.geometry.Image(depth_scaled.astype(np.float32))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=20.0, convert_rgb_to_intensity=False)
            
            self.tsdf_volume.integrate(rgbd, self.intrinsics_o3d, np.linalg.inv(self.global_pose))
            self.is_initialized = True
            return

        # Subsequent frames: estimate pose and integrate
        relative_pose, scale = self.estimate_pose_and_scale(self.prev_color_frame, resized_color_bgr, self.prev_depth_map)

        if relative_pose is not None and scale is not None:
            self.global_pose = self.global_pose @ np.linalg.inv(relative_pose)
            
            # Scale the new depth map to be consistent and integrate
            depth_scaled = new_depth_map * scale
            
            color_o3d = o3d.geometry.Image(cv2.cvtColor(resized_color_bgr, cv2.COLOR_BGR2RGB))
            depth_o3d = o3d.geometry.Image(depth_scaled.astype(np.float32))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=20.0, convert_rgb_to_intensity=False)
            
            self.tsdf_volume.integrate(rgbd, self.intrinsics_o3d, np.linalg.inv(self.global_pose))

        # Update previous frames for the next iteration
        self.prev_color_frame = resized_color_bgr
        self.prev_depth_map = new_depth_map

def main():
    parser = argparse.ArgumentParser(description="Complete 3D Reconstruction from Video using Depth Anything V2.")
    parser.add_argument('--engine', type=str, default='checkpoints/video_depth_anything_vits_17.engine', help='Path to the TensorRT engine file.')
    parser.add_argument('--video', type=str, default='video/base.mp4', help='Path to the input video file.')
    parser.add_argument('--output', type=str, default='reconstructed_scene.ply', help='Path to save the final 3D mesh.')
    args = parser.parse_args()

    # --- Camera Intrinsics (CRITICAL - Calibrate your camera for best results!) ---
    # These are example values. Use your camera's actual intrinsics.
    K_MATRIX = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])

    processor = ReconstructionProcessor(args.engine, K_MATRIX)
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- Open3D Visualizer Setup ---
    vis = o3d.visualization.Visualizer()
    vis.create_window("3D Reconstruction", 800, 600)
    mesh = o3d.geometry.TriangleMesh()
    is_mesh_added = False

    print("\nStarting reconstruction... Press 'q' in the preview window to stop early.")
    for i in tqdm(range(total_frames), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret: break

        processor.process_frame(frame)
        
        # Display the current frame
        cv2.imshow("Input Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        # Update visualization periodically (not every frame, for performance)
        if i % 20 == 0:
            new_mesh = processor.tsdf_volume.extract_triangle_mesh()
            if not new_mesh.has_vertices(): continue

            mesh.vertices = new_mesh.vertices
            mesh.triangles = new_mesh.triangles
            mesh.triangle_normals = new_mesh.triangle_normals
            mesh.vertex_colors = new_mesh.vertex_colors

            if not is_mesh_added:
                vis.add_geometry(mesh)
                is_mesh_added = True
            else:
                vis.update_geometry(mesh)
        
        vis.poll_events()
        vis.update_renderer()

    print("\nReconstruction finished. Extracting final model...")
    final_mesh = processor.tsdf_volume.extract_triangle_mesh()
    final_mesh.compute_vertex_normals()
    
    print(f"Saving final mesh to {args.output}...")
    o3d.io.write_triangle_mesh(args.output, final_mesh)

    print("Done. Displaying final model. Close the window to exit.")
    vis.run()
    vis.destroy_window()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()