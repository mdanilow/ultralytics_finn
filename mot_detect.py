import argparse
import os
from os.path import join

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.files import increment_path


# model = YOLO('ultralytics/cfg/models/v8/my_yolov8.yaml').load('yolov8n.pt')
# results = model('/home/vision/danilowi/serious_mot/ultralytics_finn/testimgs', save_features=True)
# for r in results:
#     r.show()
    # cv2.waitKey(0)
# img = results[0].plot()
# cv2.imwrite("demo.jpg", img)



parser = argparse.ArgumentParser()
# parser.add_argument('--source', type=str, default='/media/vision/1d6890f4-df75-4531-a044-f6d3d44d033d/Downloads/MOT17/train', help='source')
parser.add_argument('--source', type=str, default='/home/vision/danilowi/serious_mot/ultralytics_finn/testimgs', help='source')
parser.add_argument('--name', type=str, default='exp', help='experiment name')
parser.add_argument('--output_dir', type=str, default='runs/detect', help='output directory')
parser.add_argument('--filter_classes', type=list, default=[0], help='object classes to register')
parser.add_argument('--save_features', action="store_true", default=True, help='save intermediate features of every image into a npz')
opt = parser.parse_args()
print(opt)

model = YOLO('ultralytics/cfg/models/v8/my_yolov8.yaml').load('yolov8n.pt')
save_dir = increment_path(join(opt.output_dir, opt.name), mkdir=True)
seqnames = os.listdir(opt.source)
if 'MOT17' in opt.source:
    seqnames = [seq for seq in seqnames if 'FRCNN' in seq]
for seqname in seqnames:
    print("Processing", seqname)
    source_images_path = join(opt.source, seqname, 'img1')
    sequence_save_dir = join(save_dir, seqname)
    os.mkdir(sequence_save_dir)
    os.mkdir(join(sequence_save_dir, "features"))
    results = model(source_images_path, save_features=opt.save_features)
    det_file = ""
    for frame_idx, r in enumerate(results):
        frame_boxes = r.boxes
        for box in frame_boxes:
            if box.cls in opt.filter_classes:
                xywh = [float(el) for el in box.xywh[0]]
                conf = float(box.conf)
                line = (frame_idx + 1, -1, *xywh, conf, -1, -1, -1)
                line = [str(el) for el in line]
                det_file += ",".join(line) + "\n"
        
        if opt.save_features:
            features = [x.cpu().detach().numpy() for x in r.features]
            np.savez_compressed(join(sequence_save_dir, "features", "frame_{}.npz".format(frame_idx + 1)), *features)

    with open(join(sequence_save_dir, "det.txt"), "w") as file:
        file.write(det_file)

