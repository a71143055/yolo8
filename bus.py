from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.predict(source='bus.jpg', save=True, show=True)