import argparse
import numpy as np
import torch
import cv2

from video_depth_anything.video_depth_stream import VideoDepthAnything

def loadModel():
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    }
    model = VideoDepthAnything(**model_configs[args.encoder])
    model.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    model = model.to(DEVICE).eval()

    return model

def cv2run(model):
    cap = cv2.VideoCapture(0)
    
    while True:
        _, frame_bgr = cap.read()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            depth = model.infer_video_depth_one(frame_rgb, input_size=args.input_size, device=DEVICE)

        depth_normalized = cv2.normalize(depth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

        depth_vis = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)

        combined_display = np.hstack((frame_bgr, depth_vis))

        cv2.imshow('Video Depth Anything - Webcam (Press Q to exit)', combined_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything for Webcam')
    
    parser.add_argument('--input_size', type=int, default=200)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'])

    args = parser.parse_args()

    DEVICE = 'cuda'

    model = loadModel()

    cv2run(model)