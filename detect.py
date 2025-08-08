from ultralytics import YOLO
import cv2

# Load the official YOLOv8n model
# The library automatically handles downloading this model.
model = YOLO('yolov8n.pt')

# Initialize the webcam
# The '0' usually refers to the default built-in webcam.
# If you have multiple webcams, you might need to use 1, 2, etc.
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    print("Webcam initialized successfully. Press 'q' to quit.")
except Exception as e:
    print(f"Error: {e}")
    exit()

# Loop through the video frames
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("Webcam YOLOv8 Test", annotated_frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()
print("Webcam feed closed.")