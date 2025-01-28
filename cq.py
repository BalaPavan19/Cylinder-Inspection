import cv2
import torch
from torch.hub import load
from ultralytics import YOLO

# Load the trained YOLO model
#model = YOLO("cylinderlast.pt")

# Load the custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='cylinderlast.pt', force_reload=True)


# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect objects in the frame
    results = model(frame)

    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract class prediction
            class_id = int(box.cls[0].item())
            class_name = ["Cylinder", "Closed cap", "Opened cap"][class_id]

            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLOv8 Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()