import os
from ultralytics import YOLO
from multiprocessing import freeze_support

os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    freeze_support()
    model.train(data='datasets/data.yaml',epoches=10, imgsz=320, device=0)