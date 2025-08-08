import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import datetime
import argparse
from tqdm import tqdm
from enum import Enum
from collections import deque
import open3d as o3d # Added for 3D point cloud visualization

class TRTLoader:
    """
    Loads a TensorRT engine and handles the inference process.
    """
    def __init__(self, engine_path):
        """
        Initializes the TensorRT loader.

        Args:
            engine_path (str): The path to the TensorRT engine file.
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()

        # Allocate host and device buffers
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

        # Get input shape for preprocessing
        self.input_shape = self.engine.get_tensor_shape(self.engine[0])

    def infer(self, image):
        """
        Performs inference on a preprocessed image.

        Args:
            image (np.ndarray): The input image, preprocessed and flattened.

        Returns:
            np.ndarray: The raw output from the model.
        """
        np.copyto(self.inputs[0]['host'], image.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_v2(bindings=self.bindings)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host']

class ElevatorState(Enum):
    DOOR_CLOSED = "Door Closed"
    DOOR_OPENING = "Door Opening..."
    DOOR_OPEN = "Door Open / Stable"
    VIEW_OCCLUDED = "View Occluded!"

class DepthProcessor:
    def __init__(self, engine_path, focal_length=800.0):
        self.trt_model = TRTLoader(engine_path)
        _, _, _, self.h, self.w = self.trt_model.input_shape
        self.roi_coords = (int(self.w*0.25), int(self.h*0.1), int(self.w*0.75), int(self.h*0.9))

        self.state = ElevatorState.DOOR_CLOSED
        self.prev_depth_map = None

        self.WINDOW_SIZE = 15
        self.receding_history = deque(maxlen=self.WINDOW_SIZE)
        self.occluding_history = deque(maxlen=self.WINDOW_SIZE)

        self.RECESSION_THRESHOLD = 0.5
        self.OCCLUSION_THRESHOLD = 0.5

        roi_area = (self.roi_coords[2] - self.roi_coords[0]) * (self.roi_coords[3] - self.roi_coords[1])
        self.PIXEL_VOTE_PERCENT = 0.01
        self.PIXEL_VOTES_REQUIRED = int(roi_area * self.PIXEL_VOTE_PERCENT)
        
        self.EVENT_VOTES_REQUIRED = 8
        
        # --- 3D Point Cloud Additions ---
        self.focal_length = focal_length
        self.last_color_frame = None
        self.last_depth_map = None
        # --- End 3D Point Cloud Additions ---


    def update_elevator_state(self, current_depth_map):
        x1, y1, x2, y2 = self.roi_coords
        current_roi = current_depth_map[y1:y2, x1:x2]

        if self.prev_depth_map is None:
            self.prev_depth_map = current_depth_map
            return

        prev_roi = self.prev_depth_map[y1:y2, x1:x2]
        change_map = current_roi.astype(np.float32) - prev_roi.astype(np.float32)
        
        receding_pixels = np.sum(-change_map > self.RECESSION_THRESHOLD)
        occluding_pixels = np.sum(change_map > self.OCCLUSION_THRESHOLD)
        
        is_receding_event = receding_pixels > self.PIXEL_VOTES_REQUIRED
        is_occluding_event = occluding_pixels > self.PIXEL_VOTES_REQUIRED

        self.receding_history.append(is_receding_event)
        self.occluding_history.append(is_occluding_event)

        receding_votes = self.receding_history.count(True)
        occluding_votes = self.occluding_history.count(True)

        if occluding_votes >= self.EVENT_VOTES_REQUIRED and self.state != ElevatorState.VIEW_OCCLUDED:
            self.state_before_occlusion = self.state 
            self.state = ElevatorState.VIEW_OCCLUDED
            self.occluding_history.clear()
            self.receding_history.clear()

        if self.state == ElevatorState.VIEW_OCCLUDED:
            if occluding_votes < 2:
                self.state = self.state_before_occlusion
                self.occluding_history.clear()
                self.receding_history.clear()

        elif self.state == ElevatorState.DOOR_CLOSED:
            if receding_votes >= self.EVENT_VOTES_REQUIRED:
                self.state = ElevatorState.DOOR_OPENING
                self.receding_history.clear()

        elif self.state == ElevatorState.DOOR_OPENING:
            if receding_votes < 2:
                self.state = ElevatorState.DOOR_OPEN
                self.receding_history.clear()

        elif self.state == ElevatorState.DOOR_OPEN:
            if receding_votes >= self.EVENT_VOTES_REQUIRED:
                self.state = ElevatorState.DOOR_OPENING

        self.prev_depth_map = current_depth_map

    def process_frame(self, frame):
        original_h, original_w, _ = frame.shape
        input_image = cv2.resize(frame, (self.w, self.h))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Store resized color frame for point cloud generation
        self.last_color_frame = input_image 
        
        input_image_norm = input_image.astype(np.float32) / 255.0
        input_image_norm = input_image_norm.transpose(2, 0, 1)
        input_image_norm = np.expand_dims(input_image_norm, axis=0)

        raw_depth = self.trt_model.infer(input_image_norm)
        depth_map = raw_depth.reshape(self.h, self.w)
        
        # Store depth map for point cloud generation
        self.last_depth_map = depth_map
        
        self.update_elevator_state(depth_map.copy())

        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        depth_colormap_resized = cv2.resize(depth_colormap, (original_w, original_h))
        
        orig_x1, orig_y1, orig_x2, orig_y2 = (
            int(self.roi_coords[0] * original_w / self.w),
            int(self.roi_coords[1] * original_h / self.h),
            int(self.roi_coords[2] * original_w / self.w),
            int(self.roi_coords[3] * original_h / self.h)
        )
        
        cv2.rectangle(frame, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 2)
        cv2.rectangle(depth_colormap_resized, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 2)

        combined_display = np.hstack((frame, depth_colormap_resized))
        
        state_text = f"State: {self.state.value}"
        color = (0, 0, 255) if self.state == ElevatorState.VIEW_OCCLUDED else (0, 255, 0)
        cv2.putText(combined_display, state_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(combined_display, "Press 'p' for point cloud", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


        return combined_display

    def create_point_cloud(self):
        """
        Generates a 3D point cloud from the last processed frame.
        
        Returns:
            o3d.geometry.PointCloud: The generated point cloud, or None if data is unavailable.
        """
        if self.last_color_frame is None or self.last_depth_map is None:
            print("Color or depth data not available for point cloud generation.")
            return None

        # The model's depth is often inverse, and needs scaling. This might require tuning.
        # For 'Depth Anything', a larger depth value means closer. We invert it.
        # A scale factor is also often needed. 10.0 is a reasonable guess.
        depth_for_3d = (1.0 / (self.last_depth_map + 1e-6)) 
        depth_for_3d *= 10.0 

        color_o3d = o3d.geometry.Image(self.last_color_frame)
        depth_o3d = o3d.geometry.Image(depth_for_3d.astype(np.float32))

        # Create an RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0, # We already scaled it, so scale is 1.0 here
            depth_trunc=100.0, # Truncate distant points
            convert_rgb_to_intensity=False
        )

        # Camera intrinsic parameters (can be tuned)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=self.w,
            height=self.h,
            fx=self.focal_length,
            fy=self.focal_length,
            cx=self.w / 2,
            cy=self.h / 2
        )

        # Create point cloud from RGBD image and intrinsics
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
        
        # Flip the point cloud for a correct view (may be necessary)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        return pcd


def process_webcam(depth_processor):
    """
    Captures video from the webcam, processes it, and shows a 3D point cloud.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --- 3D Visualizer Setup ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Point Cloud", width=800, height=600)
    pcd = o3d.geometry.PointCloud()
    is_pcd_added = False
    # --- End 3D Visualizer Setup ---
    
    is_recording = False
    video_writer = None
    output_filename = ""

    print("\n--- Controls ---")
    print(" 'q': Quit")
    print(" 's': Start/Stop Recording")
    print(" 'p': Generate/Update Point Cloud in the 3D view")
    print("----------------\n")


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        combined_display = depth_processor.process_frame(frame)

        if is_recording:
            cv2.circle(combined_display, (combined_display.shape[1] - 40, 30), 10, (0, 0, 255), -1)
            if video_writer:
                video_writer.write(combined_display)

        cv2.imshow('Depth Anything V2 - Jetson', combined_display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            # Recording logic... (unchanged)
            pass
        elif key == ord('p'):
            print("Generating point cloud...")
            new_pcd = depth_processor.create_point_cloud()
            if new_pcd:
                pcd.points = new_pcd.points
                pcd.colors = new_pcd.colors
                if not is_pcd_added:
                    vis.add_geometry(pcd)
                    is_pcd_added = True
                else:
                    vis.update_geometry(pcd)
                print("Point cloud updated.")

        # Update the Open3D visualizer
        vis.poll_events()
        vis.update_renderer()

    # Cleanup
    cap.release()
    vis.destroy_window()
    cv2.destroyAllWindows()


def process_video_file(depth_processor, input_path, output_path):
    """
    Processes a video file to generate a depth estimation video.
    (Note: Interactive 3D point cloud is only available in webcam mode)
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_path}")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Point Cloud", width=800, height=600)
    pcd = o3d.geometry.PointCloud()
    is_pcd_added = False

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height))

    print(f"Processing video: {input_path}")
    print(f"Saving output to: {output_path}")
    print("Press 'q' in the preview window to stop processing early.")

    for _ in tqdm(range(total_frames), desc="Processing Frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        combined_frame = depth_processor.process_frame(frame)
        cv2.imshow('Video Processing - Press q to quit', combined_frame)
        video_writer.write(combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing stopped by user.")
            break
        elif cv2.waitKey(1) & 0xFF == ord('p'):
            print("Generating point cloud...")
            new_pcd = depth_processor.create_point_cloud()
            if new_pcd:
                pcd.points = new_pcd.points
                pcd.colors = new_pcd.colors
                if not is_pcd_added:
                    vis.add_geometry(pcd)
                    is_pcd_added = True
                else:
                    vis.update_geometry(pcd)
                print("Point cloud updated.")

        # Update the Open3D visualizer
        vis.poll_events()
        vis.update_renderer()

    print("Video processing complete.")
    cap.release()
    video_writer.release()
    vis.destroy_window()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 with TensorRT and 3D Point Cloud visualization.")
    parser.add_argument('--engine', type=str, default='./checkpoints/video_depth_anything_vits_17.engine', help='Path to the TensorRT engine file.')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--webcam', action='store_true', help='Use webcam as input for interactive 2D and 3D view.')
    group.add_argument('--video', type=str, help='Path to the input video file (2D output only).')

    parser.add_argument('--output', type=str, default='output.mp4', help='Path to save the output video file (used with --video).')
    parser.add_argument('--focal-length', type=float, default=800.0, help='Approximate focal length of the camera for 3D point cloud generation.')

    args = parser.parse_args()
    
    # Pass focal length to the processor
    processor = DepthProcessor(args.engine, args.focal_length)

    if args.webcam:
        process_webcam(processor)
    elif args.video:
        if not args.output:
            input_name = args.video.split('/')[-1].split('.')[0]
            args.output = f"{input_name}_depth.mp4"
        process_video_file(processor, args.video, args.output)


if __name__ == '__main__':
    main()