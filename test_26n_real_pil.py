import onnxruntime as ort
import numpy as np
from PIL import Image
import sys

# Load model
session = ort.InferenceSession("models/yolo26n-seg.onnx", providers=['CPUExecutionProvider'])

# Load a real image
img_path = "/home/waitangwen/Datasets/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846313.553992.png"
try:
    img0 = Image.open(img_path).convert('RGB')
except Exception as e:
    print(f"Failed to load image: {e}")
    sys.exit(1)

# Resize to 640x640 using PIL
img = img0.resize((640, 640))
img_arr = np.array(img).astype(np.float32) / 255.0

# Transpose from (H, W, C) to (C, H, W)
img_arr = np.transpose(img_arr, (2, 0, 1))

# Expand to (B, C, H, W)
img_arr = np.expand_dims(img_arr, axis=0)

# Run inference
outputs = session.run(None, {'images': img_arr})
out0 = outputs[0][0] # shape (300, 38)

# Find valid boxes
valid_boxes = [box for box in out0 if box[4] > 0.1]
print(f"Total valid boxes (conf > 0.1): {len(valid_boxes)}")

for i, box in enumerate(valid_boxes):
    cls_id = int(box[5])
    conf = box[4]
    print(f"Box {i} -> Class ID: {cls_id}, Conf: {conf:.4f}, Coords: {box[0:4]}")
