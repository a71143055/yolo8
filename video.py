import cv2
from ultralytics import YOLO
import os
os.environ['CUDA_VISIBLE_DEVICES'] = 'True'

model = YOLO('yolov8m.pt')

video_path = 'rtsp://admin:wow1989L@192.168.24.39:554/ISAPI/streaming/channels/101'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame,persist=True)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Tracking",annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()