import os
from ultralytics import YOLO
from multiprocessing import freeze_support

os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = YOLO('runs/detect/predict9/paper.jpg')