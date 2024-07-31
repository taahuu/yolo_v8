import cv2
import numpy as np
from ultralytics import YOLO, solutions

# Load the pre-trained YOLO model
model = YOLO("yolov8n.pt")
names = model.model.names

# Input video path
path = input("Enter the path: ")
cap = cv2.VideoCapture(path)

assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer setup
video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize Distance Calculation (replace with your own distance calculation logic if needed)
dist_obj = solutions.DistanceCalculation(names=names, view_img=True)

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Perform tracking
    results = model.track(im0, persist=True, show=False)
    
    # Extract bounding boxes and other details
    boxes = results[0].boxes
    centroids = []
    
    for box in boxes:
        # Extract bounding box coordinates [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Calculate centroid of the bounding box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        centroids.append((cx, cy))
        class_id = int(box.cls[0])
        if class_id == 0:  # Assuming class_id 0 is 'person'
            # Draw the bounding box and centroid
            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(im0, (cx, cy), 5, (0, 0, 255), -1)

    # Calculate distances between all pairs of centroids
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distance = calculate_distance(centroids[i], centroids[j])
            # Draw a line between centroids and display the distance
            cv2.line(im0, centroids[i], centroids[j], (255, 0, 0), 2)
            mid_point = ((centroids[i][0] + centroids[j][0]) // 2, (centroids[i][1] + centroids[j][1]) // 2)
            cv2.putText(im0, f'{distance:.2f}', mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Process and save the frame
    im0 = dist_obj.start_process(im0, results)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
