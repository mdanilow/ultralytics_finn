from ultralytics import YOLO
import cv2


model = YOLO('ultralytics/cfg/models/v8/quantyolov8.yaml').load('yolov8n.pt')

model.train(data="coco.yaml", name="first_quantyolov8", epochs=200, device=[0, 1], cache=True, plots=True, save_period=10)  # train the model
# model.train(resume=True)
# metrics = model.val(data="coco.yaml")  # evaluate model performance on the validation set
