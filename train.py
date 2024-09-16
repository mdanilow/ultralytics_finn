from ultralytics import YOLO
import cv2


model = YOLO('ultralytics/cfg/models/v8/quantyolov8.yaml').load('yolov8n.pt')
model.train(data="coco8.yaml", epochs=3)  # train the model
# metrics = model.val(data="coco.yaml")  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
img = results[0].plot()
cv2.imwrite('./trainresult.jpg', img)