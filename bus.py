from ultralytics import YOLO
model = YOLO("runs/detect/trainxx/weights/best.pt")
model.predict(source='bus.jpg', save=True, show=True)