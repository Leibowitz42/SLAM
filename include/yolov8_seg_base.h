#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "yolov8_utils.h"

/** Common interface for YOLOv8-seg backends (ONNX Runtime or TensorRT). */
class IYolov8Seg {
public:
    IYolov8Seg() {
        _className = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        };
    }
    virtual ~IYolov8Seg() = default;
    virtual bool ReadModel(const std::string& modelPath, bool isCuda = false, int cudaID = 0, bool warmUp = true) = 0;
    virtual bool OnnxDetect(cv::Mat& srcImg, std::vector<OutputParams>& output) = 0;
    virtual bool OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputParams>>& output) = 0;

    std::vector<std::string> _className;
};
