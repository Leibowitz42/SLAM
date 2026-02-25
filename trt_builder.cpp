#include <iostream>
#include "yolov8_seg_onnx.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    Yolov8SegOnnx model;
    std::cout << "Starting TensorRT engine building for " << argv[1] << std::endl;
    // Set environment flag if helpful for ONNX runtime
    bool result = model.ReadModel(argv[1], true, 0, true);
    if (result) {
        std::cout << "Engine built and loaded successfully." << std::endl;
    } else {
        std::cout << "Failed to load model." << std::endl;
    }
    return 0;
}
