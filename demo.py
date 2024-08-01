import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.nn.modules import C2f


# Load a model
model = YOLO("yolov8n.yaml").load('yolov8n.pt')  # load an official model


for m in model.modules():
    if(isinstance(m, C2f)):
        m.forward = m.forward_split


# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
img = results[0].plot()
cv2.imwrite('./result.jpg', img)