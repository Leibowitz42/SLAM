import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("models/yolo26n-seg.onnx", providers=['CPUExecutionProvider'])

img = np.zeros((640, 640, 3), dtype=np.uint8)
img[100:200, 100:200, :] = 255

img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

outputs = session.run(None, {'images': img})
out0 = outputs[0][0] # shape (300, 38)
out1 = outputs[1][0] # shape (32, 160, 160)

valid_boxes = [box for box in out0 if np.sum(np.abs(box)) > 0]
print(f"Total valid boxes: {len(valid_boxes)}")
if len(valid_boxes) > 0:
    for i in range(min(2, len(valid_boxes))):
        box = valid_boxes[i]
        print(f"Box {i}:")
        print(f"  Coords: {box[0:4]}")
        print(f"  Conf: {box[4]}")
        print(f"  Class: {box[5]}")
        print(f"  Mask features: {box[6:10]} ...")
