import argparse
import numpy as np
import torch
import time
import cv2

from video_depth_anything.video_depth_stream import VideoDepthAnything

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything for Webcam')
    
    parser.add_argument('--cam_index', type=int, default=0, help='Index of the webcam to use')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=320, help='Maximum resolution for webcam feed')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'])
    parser.add_argument('--fp32', action='store_true', help='Use torch.float32 for inference, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='Output grayscale depth map instead of color')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    cap = cv2.VideoCapture(args.cam_index)
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam at index {args.cam_index}")

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    target_width, target_height = original_width, original_height
    if args.max_res > 0 and max(original_width, original_height) > args.max_res:
        scale = args.max_res / max(original_width, original_height)
        target_width = round(original_width * scale)
        target_height = round(original_height * scale)

    prev_time = 0
    
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            break

        if (target_width, target_height) != (original_width, original_height):
            frame_bgr = cv2.resize(frame_bgr, (target_width, target_height))
            
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            depth = video_depth_anything.infer_video_depth_one(frame_rgb, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

        depth_normalized = cv2.normalize(depth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        if args.grayscale:
            depth_vis = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
        else:
            depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
            
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(frame_bgr, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        combined_display = np.hstack((frame_bgr, depth_vis))
        cv2.imshow('Video Depth Anything - Webcam (Press Q to exit)', combined_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()