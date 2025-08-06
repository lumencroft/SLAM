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
    DOOR_OPENING = "Door Opening..."      # 문이 멀어지는 중
    DOOR_OPEN = "Door Open / Stable"
    VIEW_OCCLUDED = "View Occluded!" # 시야가 가까운 물체에 의해 가려짐

class DepthProcessor:
    def __init__(self, engine_path):
        self.trt_model = TRTLoader(engine_path)
        _, _, _, self.h, self.w = self.trt_model.input_shape
        self.roi_coords = (int(self.w*0.25), int(self.h*0.1), int(self.w*0.75), int(self.h*0.9))

        self.state = ElevatorState.DOOR_CLOSED
        self.prev_depth_map = None

        # --- 슬라이딩 윈도우 및 임계값 재설정 ---
        self.WINDOW_SIZE = 15
        self.receding_history = deque(maxlen=self.WINDOW_SIZE) # 멀어짐 기록
        self.occluding_history = deque(maxlen=self.WINDOW_SIZE)# 가까워짐 기록

        # 뎁스 값의 변화량 임계값 (1.0 = 1미터 차이. 모델의 출력 스케일에 따라 조정)
        self.RECESSION_THRESHOLD = 0.5  # 0.5m 이상 멀어진 변화를 감지
        self.OCCLUSION_THRESHOLD = 0.8  # 0.3m 이상 가까워진 변화를 감지

        # ROI 영역 내에서 몇 %의 픽셀이 변해야 이벤트로 판단할지 결정
        roi_area = (self.roi_coords[2] - self.roi_coords[0]) * (self.roi_coords[3] - self.roi_coords[1])
        self.PIXEL_VOTE_PERCENT = 0.01 # ROI 영역의 5% 이상에서 변화가 감지되어야 함
        self.PIXEL_VOTES_REQUIRED = int(roi_area * self.PIXEL_VOTE_PERCENT)
        
        # 윈도우 내에서 몇 번의 이벤트가 감지되어야 상태를 바꿀지 결정
        self.EVENT_VOTES_REQUIRED = 8

    def update_elevator_state(self, current_depth_map):
        x1, y1, x2, y2 = self.roi_coords
        current_roi = current_depth_map[y1:y2, x1:x2]

        if self.prev_depth_map is None:
            self.prev_depth_map = current_depth_map
            return

        prev_roi = self.prev_depth_map[y1:y2, x1:x2]

        # --- 변화의 '방향' 계산 ---
        change_map = current_roi.astype(np.float32) - prev_roi.astype(np.float32)

        # ================================================================= #
        # ## <<< 핵심 수정 사항: 모델 특성에 맞게 계산 로직을 맞바꿈 ## #
        # ================================================================= #
        
        # receding (멀어짐): 뎁스 값 '감소' -> change_map이 음수 -> -change_map이 양수가 됨
        receding_pixels = np.sum(-change_map > self.RECESSION_THRESHOLD)
        
        # occluding (가까워짐): 뎁스 값 '증가' -> change_map이 양수가 됨
        occluding_pixels = np.sum(change_map > self.OCCLUSION_THRESHOLD)

        # 디버깅이 필요하면 아래 주석을 해제하세요.
        # print(f"Receding Pixels: {receding_pixels}, Occluding Pixels: {occluding_pixels}")
        
        is_receding_event = receding_pixels > self.PIXEL_VOTES_REQUIRED
        is_occluding_event = occluding_pixels > self.PIXEL_VOTES_REQUIRED

        self.receding_history.append(is_receding_event)
        self.occluding_history.append(is_occluding_event)

        receding_votes = self.receding_history.count(True)
        occluding_votes = self.occluding_history.count(True)

        # --- 지능적인 상태 머신 (이전 제안 포함) ---

        # 1. 시야 가림(Occlusion)을 최우선으로 처리
        if occluding_votes >= self.EVENT_VOTES_REQUIRED and self.state != ElevatorState.VIEW_OCCLUDED:
            self.state_before_occlusion = self.state 
            self.state = ElevatorState.VIEW_OCCLUDED
            self.occluding_history.clear()
            self.receding_history.clear()

        # 2. 상태별 로직 처리
        if self.state == ElevatorState.VIEW_OCCLUDED:
            if occluding_votes < 2: # STABLE_VOTES_THRESHOLD 값으로 2를 사용
                self.state = self.state_before_occlusion
                self.occluding_history.clear()
                self.receding_history.clear()

        elif self.state == ElevatorState.DOOR_CLOSED:
            if receding_votes >= self.EVENT_VOTES_REQUIRED:
                self.state = ElevatorState.DOOR_OPENING
                self.receding_history.clear()

        elif self.state == ElevatorState.DOOR_OPENING:
            if receding_votes < 2: # STABLE_VOTES_THRESHOLD 값으로 2를 사용
                self.state = ElevatorState.DOOR_OPEN
                self.receding_history.clear()

        elif self.state == ElevatorState.DOOR_OPEN:
            if receding_votes >= self.EVENT_VOTES_REQUIRED:
                self.state = ElevatorState.DOOR_OPENING
            # (필요시) 문 닫힘 로직을 여기에 occluding_votes를 사용하여 추가 가능

        self.prev_depth_map = current_depth_map

    # process_frame 메서드는 이전과 동일하게 유지됩니다.
    def process_frame(self, frame):
        original_h, original_w, _ = frame.shape
        input_image = cv2.resize(frame, (self.w, self.h))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32) / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)

        raw_depth = self.trt_model.infer(input_image)
        depth_map = raw_depth.reshape(self.h, self.w)
        
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

        return combined_display

def process_webcam(depth_processor):
    """
    Captures video from the webcam and processes it in real-time.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    is_recording = False
    video_writer = None
    output_filename = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        combined_display = depth_processor.process_frame(frame)

        if is_recording:
            cv2.circle(combined_display, (combined_display.shape[1] - 40, 30), 10, (0, 0, 255), -1)
            if video_writer:
                video_writer.write(combined_display)

        cv2.imshow('Depth Anything V2 - Jetson (q: quit, s: rec)', combined_display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            if not is_recording:
                is_recording = True
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"recording_{timestamp}.mp4"
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_h, frame_w, _ = combined_display.shape
                video_writer = cv2.VideoWriter(output_filename, fourcc, 10.0, (frame_w, frame_h))
                print(f"Recording started. Saving to: {output_filename}")
            else:
                is_recording = False
                if video_writer:
                    video_writer.release()
                video_writer = None
                print(f"Recording stopped. File saved as '{output_filename}'.")

    if is_recording and video_writer:
        video_writer.release()

    cap.release()
    cv2.destroyAllWindows()

def process_video_file(depth_processor, input_path, output_path):
    """
    Processes a video file to generate a depth estimation video.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_path}")
        return

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

    print("Video processing complete.")
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 with TensorRT on webcam or video file.")
    parser.add_argument('--engine', type=str, default='./checkpoints/video_depth_anything_vits_17.engine', help='Path to the TensorRT engine file.')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--webcam', action='store_true', help='Use webcam as input.')
    group.add_argument('--video', type=str, help='Path to the input video file.')

    parser.add_argument('--output', type=str, default='output.mp4', help='Path to save the output video file (used with --video).')

    args = parser.parse_args()
    processor = DepthProcessor(args.engine)

    if args.webcam:
        process_webcam(processor)
    elif args.video:
        if not args.output:
            input_name = args.video.split('/')[-1].split('.')[0]
            args.output = f"{input_name}_depth.mp4"
        process_video_file(processor, args.video, args.output)

if __name__ == '__main__':
    main()