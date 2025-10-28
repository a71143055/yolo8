import os
from ultralytics import YOLO
from multiprocessing import freeze_support

os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = YOLO('yolov8m.pt')

if __name__ == '__main__':
    freeze_support()
    model.train(data='fashion-mnist', epoch=10, imgsz=320, device=0)