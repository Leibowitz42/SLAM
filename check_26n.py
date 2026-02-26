import onnxruntime as ort
import sys

try:
    print("Inspecting models/yolo26n-seg.onnx")
    session = ort.InferenceSession("models/yolo26n-seg.onnx", providers=['CPUExecutionProvider'])
    for i, input in enumerate(session.get_inputs()):
        print(f"Input {i}: name={input.name}, shape={input.shape}, type={input.type}")
    for i, output in enumerate(session.get_outputs()):
        print(f"Output {i}: name={output.name}, shape={output.shape}, type={output.type}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
