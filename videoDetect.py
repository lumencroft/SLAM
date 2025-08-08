from ultralytics import YOLO
import cv2

# Load the official YOLOv8n model
model = YOLO('yolov8n.pt')

# --- CHANGE IS HERE ---
# Path to your local video file
video_path = './video/base.mp4'

try:
    # Initialize the video capture from the file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    print("Video file loaded successfully. Press 'q' to quit early.")
except Exception as e:
    print(f"Error: {e}")
    exit()

# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # If 'ret' is False, it means the video has ended
    if not ret:
        print("End of video reached.")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Video Test", annotated_frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close the display window
cap.release()
cv2.destroyAllWindows()
print("Processing complete.")