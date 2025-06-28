
import torch
import cv2
import numpy as np

# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define vehicle classes from COCO dataset
vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']

# Initialize video capture (0 for webcam or path to video file)
cap = cv2.VideoCapture("traffic_sample.mp4")  # Replace with 0 for webcam

vehicle_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Convert results to pandas dataframe
    df = results.pandas().xyxy[0]

    # Filter only vehicle classes
    vehicles = df[df['name'].isin(vehicle_classes)]

    # Draw bounding boxes and labels
    for _, row in vehicles.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = f"{row['name']} {row['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show vehicle count on frame
    cv2.putText(frame, f"Vehicles detected: {len(vehicles)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display result
    cv2.imshow('TrafficTelligence - Vehicle Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
