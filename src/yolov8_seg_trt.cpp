#include "yolov8_seg_trt.h"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <numeric>

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

namespace {
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, char const* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << std::endl;
    }
};
static TrtLogger gTrtLogger;

inline size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i)
        v *= static_cast<size_t>(d.d[i] > 0 ? d.d[i] : 1);
    return v;
}

inline size_t elementSize(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        default: return 4;
    }
}
} // namespace

Yolov8SegTrt::~Yolov8SegTrt() {
    gpuPreprocessDestroy(_gpuCtx);
    if (_stream) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(_stream));
        cudaStreamDestroy(static_cast<cudaStream_t>(_stream));
        _stream = nullptr;
    }
    if (_inputDevice)  { cudaFree(_inputDevice);   _inputDevice = nullptr; }
    if (_output0Device){ cudaFree(_output0Device);  _output0Device = nullptr; }
    if (_output1Device){ cudaFree(_output1Device);  _output1Device = nullptr; }
    if (_output0Host)  { delete[] _output0Host;    _output0Host = nullptr; }
    if (_output1Host) { delete[] _output1Host;    _output1Host = nullptr; }
    if (_context) {
        delete static_cast<nvinfer1::IExecutionContext*>(_context);
        _context = nullptr;
    }
    if (_engine) {
        delete static_cast<nvinfer1::ICudaEngine*>(_engine);
        _engine = nullptr;
    }
    if (_runtime) {
        delete static_cast<nvinfer1::IRuntime*>(_runtime);
        _runtime = nullptr;
    }
}

bool Yolov8SegTrt::ReadModel(const std::string& modelPath, bool isCuda, int cudaID, bool warmUp) {
    (void)isCuda;
    (void)cudaID;
    if (!CheckModelPath(modelPath))
        return false;

    std::ifstream f(modelPath, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open engine file: " << modelPath << std::endl;
        return false;
    }
    f.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(size);
    if (!f.read(blob.data(), size)) {
        std::cerr << "Failed to read engine file." << std::endl;
        return false;
    }
    f.close();

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gTrtLogger);
    if (!runtime) {
        std::cerr << "createInferRuntime failed." << std::endl;
        return false;
    }
    _runtime = runtime;

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(blob.data(), size);
    if (!engine) {
        std::cerr << "deserializeCudaEngine failed." << std::endl;
        return false;
    }
    _engine = engine;

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "createExecutionContext failed." << std::endl;
        return false;
    }
    _context = context;

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "cudaStreamCreate failed." << std::endl;
        return false;
    }
    _stream = stream;

    // Tensor names for YOLOv8 seg (from ONNX export)
    const char* inputName  = "images";
    const char* output0Name = "output0";
    const char* output1Name = "output1";

    nvinfer1::Dims inDims  = context->getTensorShape(inputName);
    nvinfer1::Dims out0Dims = context->getTensorShape(output0Name);
    nvinfer1::Dims out1Dims = context->getTensorShape(output1Name);

    size_t es = elementSize(engine->getTensorDataType(inputName));
    _inputSize  = volume(inDims)  * es;
    _output0Size = volume(out0Dims) * elementSize(engine->getTensorDataType(output0Name));
    _output1Size = volume(out1Dims) * elementSize(engine->getTensorDataType(output1Name));

    _output0Shape.resize(out0Dims.nbDims);
    for (int i = 0; i < out0Dims.nbDims; ++i) _output0Shape[i] = out0Dims.d[i];
    _output1Shape.resize(out1Dims.nbDims);
    for (int i = 0; i < out1Dims.nbDims; ++i) _output1Shape[i] = out1Dims.d[i];

    // [bs, 116, 8400] -> not nms_embedded; [bs, 8400, 38] -> nms_embedded
    _isNmsEmbedded = (_output0Shape.size() == 3 && _output0Shape[2] == 4 + 1 + 1 + static_cast<int64_t>(_output1Shape[1]));

    if (cudaMalloc(&_inputDevice, _inputSize) != cudaSuccess ||
        cudaMalloc(&_output0Device, _output0Size) != cudaSuccess ||
        cudaMalloc(&_output1Device, _output1Size) != cudaSuccess) {
        std::cerr << "cudaMalloc failed." << std::endl;
        return false;
    }
    _output0Host = new float[_output0Size / sizeof(float)];
    _output1Host = new float[_output1Size / sizeof(float)];

    if (!context->setInputTensorAddress(inputName, _inputDevice) ||
        !context->setTensorAddress(output0Name, _output0Device) ||
        !context->setTensorAddress(output1Name, _output1Device)) {
        std::cerr << "setTensorAddress failed." << std::endl;
        return false;
    }

    if (warmUp) {
        std::cout << "TensorRT warm up..." << std::endl;
        std::vector<cv::Mat> dummy(1, cv::Mat(640, 640, CV_8UC3));
        std::vector<std::vector<OutputParams>> out;
        OnnxBatchDetect(dummy, out);
        std::cout << "TensorRT warm up done." << std::endl;
    }
    return true;
}

int Yolov8SegTrt::PreProcessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params) {
    outSrcImgs.clear();
    cv::Size input_size(_netWidth, _netHeight);
    for (size_t i = 0; i < srcImgs.size(); ++i) {
        cv::Mat temp_img = srcImgs[i];
        cv::Vec4d temp_param(1, 1, 0, 0);
        if (temp_img.size() != input_size) {
            cv::Mat borderImg;
            LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
            outSrcImgs.push_back(borderImg);
            params.push_back(temp_param);
        } else {
            outSrcImgs.push_back(temp_img);
            params.push_back(temp_param);
        }
    }
    int lack = static_cast<int>(srcImgs.size() % _batchSize);
    if (lack != 0) {
        for (int i = 0; i < lack; ++i) {
            outSrcImgs.push_back(cv::Mat::zeros(input_size, CV_8UC3));
            params.push_back(cv::Vec4d(1, 1, 0, 0));
        }
    }
    return 0;
}

bool Yolov8SegTrt::OnnxDetect(cv::Mat& srcImg, std::vector<OutputParams>& output) {
    std::vector<cv::Mat> input_data = { srcImg };
    std::vector<std::vector<OutputParams>> batch_out;
    if (!OnnxBatchDetect(input_data, batch_out))
        return false;
    output = batch_out[0];
    return true;
}

bool Yolov8SegTrt::OnnxDetectGpu(cv::Mat& srcImg, std::vector<OutputParams>& output) {
    if(!_gpuPreprocessInited){
        gpuPreprocessInit(_gpuCtx, srcImg.cols, srcImg.rows, _netWidth, _netHeight);
        _gpuPreprocessInited = true;
    }
    
    cudaStream_t stream = static_cast<cudaStream_t>(_stream);

    // EVOLUTION 2.0: True UMA Zero-Copy. The raw image writes its output directly into TensorRT's bindings buffer!
    // No D2D copy overhead. No memory controller congestion.
    gpuPreprocessExecuteDirect(_gpuCtx, srcImg.data, srcImg.cols, srcImg.rows, (int)srcImg.step, 
                               static_cast<float*>(_inputDevice));
                               
    // Sync the preprocessing stream strictly before enqueueing TRT graph on its own stream
    cudaStreamSynchronize(static_cast<cudaStream_t>(_gpuCtx.stream));

    nvinfer1::IExecutionContext* ctx = static_cast<nvinfer1::IExecutionContext*>(_context);
    if (!ctx->enqueueV3(stream)) return false;
    
    if (cudaMemcpyAsync(_output0Host, _output0Device, _output0Size, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(_output1Host, _output1Device, _output1Size, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return false;
    if (cudaStreamSynchronize(stream) != cudaSuccess)
        return false;

    // Post processing
    int64_t one_output_length = VectorProduct(_output0Shape) / _output0Shape[0];
    int net_width = _isNmsEmbedded ? static_cast<int>(_output0Shape[2]) : static_cast<int>(_output0Shape[1]);
    int score_array_length = net_width - 4 - static_cast<int>(_output1Shape[1]);
    std::vector<int> mask_protos_shape = { 1, static_cast<int>(_output1Shape[1]), static_cast<int>(_output1Shape[2]), static_cast<int>(_output1Shape[3]) };
    int mask_protos_length = static_cast<int>(VectorProduct(_output1Shape));
    
    cv::Vec4d gpu_params( _gpuCtx.ratio, _gpuCtx.ratio, _gpuCtx.pad_w, _gpuCtx.pad_h );
    float* all_data = _output0Host;
    
    cv::Mat output0;
    if (_isNmsEmbedded)
        output0 = cv::Mat(cv::Size(static_cast<int>(_output0Shape[2]), static_cast<int>(_output0Shape[1])), CV_32F, all_data).clone();
    else
        output0 = cv::Mat(cv::Size(static_cast<int>(_output0Shape[2]), static_cast<int>(_output0Shape[1])), CV_32F, all_data).t();

    float* pdata = reinterpret_cast<float*>(output0.data);
    int rows = output0.rows;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> picked_proposals;

    for (int r = 0; r < rows; ++r) {
        cv::Point classIdPoint;
        double max_class_score;
        std::vector<float> temp_proto;
        if (_isNmsEmbedded) {
            max_class_score = pdata[4];
            classIdPoint.x = static_cast<int>(pdata[5]);
            temp_proto = std::vector<float>(pdata + 6, pdata + net_width);
        } else {
            cv::Mat scores(1, score_array_length, CV_32F, pdata + 4);
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);
            max_class_score = static_cast<float>(max_class_score);
            temp_proto = std::vector<float>(pdata + 4 + score_array_length, pdata + net_width);
        }
        if (max_class_score >= _classThreshold) {
            picked_proposals.push_back(temp_proto);
            int left, top, width, height;
            if (_isNmsEmbedded) {
                left   = MAX(static_cast<int>((pdata[0] - gpu_params[2]) / gpu_params[0] + 0.5), 0);
                top    = MAX(static_cast<int>((pdata[1] - gpu_params[3]) / gpu_params[1] + 0.5), 0);
                int right  = MAX(static_cast<int>((pdata[2] - gpu_params[2]) / gpu_params[0] + 0.5), 0);
                int bottom = MAX(static_cast<int>((pdata[3] - gpu_params[3]) / gpu_params[1] + 0.5), 0);
                width  = right - left;
                height = bottom - top;
            } else {
                float x = (pdata[0] - gpu_params[2]) / gpu_params[0];
                float y = (pdata[1] - gpu_params[3]) / gpu_params[1];
                float w = pdata[2] / gpu_params[0];
                float h = pdata[3] / gpu_params[1];
                left   = MAX(static_cast<int>(x - 0.5 * w + 0.5), 0);
                top    = MAX(static_cast<int>(y - 0.5 * h + 0.5), 0);
                width  = static_cast<int>(w + 0.5);
                height = static_cast<int>(h + 0.5);
            }
            class_ids.push_back(classIdPoint.x);
            confidences.push_back(static_cast<float>(max_class_score));
            boxes.push_back(cv::Rect(left, top, width, height));
        }
        pdata += net_width;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
    cv::Rect holeImgRect(0, 0, srcImg.cols, srcImg.rows);
    std::vector<OutputParams> temp_output;
    std::vector<std::vector<float>> temp_mask_proposals;
    for (int i : nms_result) {
        OutputParams result;
        result.id = class_ids[i];
        result.confidence = confidences[i];
        result.box = boxes[i] & holeImgRect;
        temp_mask_proposals.push_back(picked_proposals[i]);
        temp_output.push_back(result);
    }

    MaskParams mask_params;
    mask_params.params = gpu_params;
    mask_params.srcImgShape = srcImg.size();
    mask_params.netHeight = _netHeight;
    mask_params.netWidth = _netWidth;
    mask_params.maskThreshold = _maskThreshold;
    cv::Mat mask_protos(mask_protos_shape, CV_32F, _output1Host);
    for (size_t i = 0; i < temp_mask_proposals.size(); ++i)
        GetMask2(cv::Mat(temp_mask_proposals[i]).t(), mask_protos, temp_output[i], mask_params);
    output = temp_output;
    return !output.empty();
}

bool Yolov8SegTrt::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputParams>>& output) {
    using namespace cv;
    std::vector<Vec4d> params;
    std::vector<Mat> input_images;
    PreProcessing(srcImgs, input_images, params);
    cv::Size input_size(_netWidth, _netHeight);
    Mat blob = cv::dnn::blobFromImages(input_images, 1.0 / 255.0, input_size, Scalar(0, 0, 0), true, false);

    cudaStream_t stream = static_cast<cudaStream_t>(_stream);
    if (cudaMemcpyAsync(_inputDevice, blob.ptr<float>(), _inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return false;

    nvinfer1::IExecutionContext* ctx = static_cast<nvinfer1::IExecutionContext*>(_context);
    if (!ctx->enqueueV3(stream)) {
        std::cerr << "enqueueV3 failed." << std::endl;
        return false;
    }
    if (cudaMemcpyAsync(_output0Host, _output0Device, _output0Size, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(_output1Host, _output1Device, _output1Size, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return false;
    if (cudaStreamSynchronize(stream) != cudaSuccess)
        return false;

    // Postprocess (same layout as ONNX)
    int64_t one_output_length = VectorProduct(_output0Shape) / _output0Shape[0];
    int net_width = _isNmsEmbedded ? static_cast<int>(_output0Shape[2]) : static_cast<int>(_output0Shape[1]);
    int score_array_length = net_width - 4 - static_cast<int>(_output1Shape[1]);
    std::vector<int> mask_protos_shape = { 1, static_cast<int>(_output1Shape[1]), static_cast<int>(_output1Shape[2]), static_cast<int>(_output1Shape[3]) };
    int mask_protos_length = static_cast<int>(VectorProduct(_output1Shape));

    float* all_data = _output0Host;
    for (size_t img_index = 0; img_index < srcImgs.size(); ++img_index) {
        Mat output0;
        if (_isNmsEmbedded)
            output0 = Mat(Size(static_cast<int>(_output0Shape[2]), static_cast<int>(_output0Shape[1])), CV_32F, all_data).clone();
        else
            output0 = Mat(Size(static_cast<int>(_output0Shape[2]), static_cast<int>(_output0Shape[1])), CV_32F, all_data).t();
        all_data += one_output_length;

        float* pdata = reinterpret_cast<float*>(output0.data);
        int rows = output0.rows;
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<Rect> boxes;
        std::vector<std::vector<float>> picked_proposals;

        for (int r = 0; r < rows; ++r) {
            Point classIdPoint;
            double max_class_score;
            std::vector<float> temp_proto;
            if (_isNmsEmbedded) {
                max_class_score = pdata[4];
                classIdPoint.x = static_cast<int>(pdata[5]);
                temp_proto = std::vector<float>(pdata + 6, pdata + net_width);
            } else {
                Mat scores(1, score_array_length, CV_32F, pdata + 4);
                minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);
                max_class_score = static_cast<float>(max_class_score);
                temp_proto = std::vector<float>(pdata + 4 + score_array_length, pdata + net_width);
            }
            if (max_class_score >= _classThreshold) {
                picked_proposals.push_back(temp_proto);
                int left, top, width, height;
                if (_isNmsEmbedded) {
                    left   = MAX(static_cast<int>((pdata[0] - params[img_index][2]) / params[img_index][0] + 0.5), 0);
                    top    = MAX(static_cast<int>((pdata[1] - params[img_index][3]) / params[img_index][1] + 0.5), 0);
                    int right  = MAX(static_cast<int>((pdata[2] - params[img_index][2]) / params[img_index][0] + 0.5), 0);
                    int bottom = MAX(static_cast<int>((pdata[3] - params[img_index][3]) / params[img_index][1] + 0.5), 0);
                    width  = right - left;
                    height = bottom - top;
                } else {
                    float x = (pdata[0] - params[img_index][2]) / params[img_index][0];
                    float y = (pdata[1] - params[img_index][3]) / params[img_index][1];
                    float w = pdata[2] / params[img_index][0];
                    float h = pdata[3] / params[img_index][1];
                    left   = MAX(static_cast<int>(x - 0.5 * w + 0.5), 0);
                    top    = MAX(static_cast<int>(y - 0.5 * h + 0.5), 0);
                    width  = static_cast<int>(w + 0.5);
                    height = static_cast<int>(h + 0.5);
                }
                class_ids.push_back(classIdPoint.x);
                confidences.push_back(static_cast<float>(max_class_score));
                boxes.push_back(Rect(left, top, width, height));
            }
            pdata += net_width;
        }

        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
        cv::Rect holeImgRect(0, 0, srcImgs[img_index].cols, srcImgs[img_index].rows);
        std::vector<OutputParams> temp_output;
        std::vector<std::vector<float>> temp_mask_proposals;
        for (int i : nms_result) {
            OutputParams result;
            result.id = class_ids[i];
            result.confidence = confidences[i];
            result.box = boxes[i] & holeImgRect;
            temp_mask_proposals.push_back(picked_proposals[i]);
            temp_output.push_back(result);
        }

        MaskParams mask_params;
        mask_params.params = params[img_index];
        mask_params.srcImgShape = srcImgs[img_index].size();
        mask_params.netHeight = _netHeight;
        mask_params.netWidth = _netWidth;
        mask_params.maskThreshold = _maskThreshold;
        Mat mask_protos(mask_protos_shape, CV_32F, _output1Host + img_index * mask_protos_length);
        for (size_t i = 0; i < temp_mask_proposals.size(); ++i)
            GetMask2(Mat(temp_mask_proposals[i]).t(), mask_protos, temp_output[i], mask_params);
        output.push_back(temp_output);
    }
    return !output.empty();
}
