#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "yolov8_utils.h"

/**
 * @brief Unified interface for YOLOv8 Segmentation Models
 * This interface allows swapping different backend implementations
 * (e.g. ONNX Runtime, pure TensorRT, LibTorch) interchangeably
 * on different platforms (PC / Jetson).
 */
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

    std::vector<std::string> _className;

    /**
     * @brief Load model
     * @param modelPath Model file path
     * @param isCuda Use CUDA/GPU backend
     * @param cudaID GPU device ID
     * @param warmUp Run a warmup inference
     * @return true if successful
     */
    virtual bool ReadModel(const std::string& modelPath, bool isCuda = false, int cudaID = 0, bool warmUp = true) = 0;

    /**
     * @brief Detect from single image (legacy CPU preprocessing)
     * @param srcImg input image
     * @param output detected objects and masks
     */
    virtual bool OnnxDetect(cv::Mat& srcImg, std::vector<OutputParams>& output) = 0;

    /**
     * @brief Detect with full GPU preprocessing pipeline
     * @param srcImg input image (transferred to GPU directly)
     * @param output detected objects and masks
     */
    virtual bool OnnxDetectGpu(cv::Mat& srcImg, std::vector<OutputParams>& output) = 0;

    /**
     * @brief Batch detect (legacy CPU preprocessing)
     */
    virtual bool OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputParams>>& output) = 0;
};
