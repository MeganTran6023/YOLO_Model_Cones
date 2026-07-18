from ultralytics import YOLO
import cv2

# Load your trained model
# Replace 'path/to/best.pt' with the actual path to your trained weights
model = YOLO('runs/detect/runs/train/exp/weights/best.pt')

# Open the live camera feed (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the result
    cv2.imshow("YOLO Real-Time Inference", annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()