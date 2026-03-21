#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "IYolov8Seg.h"
#include "yolov8_utils.h"
#include "gpu_preprocess.h"

class Yolov8SegTrt : public IYolov8Seg {
public:
    Yolov8SegTrt() = default;
    ~Yolov8SegTrt() override;

    bool ReadModel(const std::string& modelPath, bool isCuda = false, int cudaID = 0, bool warmUp = true) override;
    bool OnnxDetect(cv::Mat& srcImg, std::vector<OutputParams>& output) override;
    bool OnnxDetectGpu(cv::Mat& srcImg, std::vector<OutputParams>& output) override;
    bool OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputParams>>& output) override;

private:
    template <typename T>
    T VectorProduct(const std::vector<T>& v) {
        return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
    }
    int PreProcessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params);

    static const int _netWidth = 640;
    static const int _netHeight = 640;
    int _batchSize = 1;
    float _classThreshold = 0.5f;
    float _nmsThreshold = 0.45f;
    float _maskThreshold = 0.5f;

    // TensorRT
    void* _runtime = nullptr;
    void* _engine = nullptr;
    void* _context = nullptr;
    void* _stream = nullptr;
    void* _inputDevice = nullptr;
    void* _output0Device = nullptr;
    void* _output1Device = nullptr;
    float* _output0Host = nullptr;
    float* _output1Host = nullptr;
    int _inputIndex = -1;
    int _output0Index = -1;
    int _output1Index = -1;
    size_t _inputSize = 0;
    size_t _output0Size = 0;
    size_t _output1Size = 0;
    std::vector<int64_t> _output0Shape;
    std::vector<int64_t> _output1Shape;
    bool _isNmsEmbedded = false;

    // GPU Preprocessing context
    GpuPreprocessContext _gpuCtx;
    bool _gpuPreprocessInited = false;
};
