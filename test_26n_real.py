import onnxruntime as ort
import numpy as np
import cv2
import sys

# Load model
session = ort.InferenceSession("models/yolo26n-seg.onnx", providers=['CPUExecutionProvider'])

# Load a real image
img_path = "/home/waitangwen/Datasets/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846313.553992.png"
img0 = cv2.imread(img_path)
if img0 is None:
    print(f"Failed to load image: {img_path}")
    sys.exit(1)

# Resize and preprocess according to YOLOv8
img = cv2.resize(img0, (640, 640))
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

# Run inference
outputs = session.run(None, {'images': img})
out0 = outputs[0][0] # shape (300, 38)

# Find valid boxes
valid_boxes = [box for box in out0 if box[4] > 0.1]
print(f"Total valid boxes (conf > 0.1): {len(valid_boxes)}")

import json
for i, box in enumerate(valid_boxes):
    cls_id = int(box[5])
    conf = box[4]
    print(f"Box {i} -> Class ID: {cls_id}, Conf: {conf:.4f}, Coords: {box[0:4]}")

