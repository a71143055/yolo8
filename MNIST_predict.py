import os
from ultralytics import YOLO
from multiprocessing import freeze_support

os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = YOLO('yolov8n-cls.pt')

if __name__ == '__main__':
    freeze_support()
    results = model('https://ultralytics.com/images/bus.jpg')