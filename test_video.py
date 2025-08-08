import cv2
import os

video_path = './video/man.mp4'

print("--- OpenCV Video Test ---")
print(f"Checking if file exists at: {video_path}")

if not os.path.exists(video_path):
    print(f"❌ FAILURE: The file does not exist at the specified path.")
else:
    print(f"✅ File found. Attempting to open with OpenCV...")
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        print("✅ SUCCESS: OpenCV successfully opened the video file.")
        ret, frame = cap.read()
        if ret:
            print(f"    -> Successfully read a frame with resolution {frame.shape[1]}x{frame.shape[0]}")
        else:
            print("    -> ⚠️ WARNING: Could open the file, but failed to read the first frame.")
    else:
        print("❌ FAILURE: OpenCV could NOT open the video file.")
        print("    -> This is the problem. It is almost certainly due to missing video codecs.")
        print("    -> To fix this, run the following command in your terminal:")
        print("\n        sudo apt-get update && sudo apt-get install -y ffmpeg libsm6 libxext6\n")

    cap.release()
print("--- Test Complete ---")