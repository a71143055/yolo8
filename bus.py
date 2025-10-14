from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")
model.predict(source='bus.jpg', save=True, show=True)