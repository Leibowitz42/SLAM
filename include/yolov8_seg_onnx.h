/*
* This file is part of YDM-SLAM
* Author: Balveer Singh
* GitHub: https://github.com/balveersinghyt/YDM-SLAM
*/
#pragma once
#include <iostream>
#include<memory>
#include <opencv2/opencv.hpp>
#include "yolov8_utils.h"
#include "gpu_preprocess.h"
#include "IYolov8Seg.h"
#include<onnxruntime_cxx_api.h>
//#include <tensorrt_provider_factory.h>  //if use OrtTensorRTProviderOptionsV2
//#include <onnxruntime_c_api.h>

class Yolov8SegOnnx : public IYolov8Seg {
public:
	Yolov8SegOnnx() :
		_OrtMemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput)),
		_OrtMemoryInfoCuda(Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault))
	{};
	~Yolov8SegOnnx() {
		gpuPreprocessDestroy(_gpuCtx);
		if (_OrtSession != nullptr)
			delete _OrtSession;
	};// delete _OrtMemoryInfo;


public:
	/** \brief Read onnx-model
	* \param[in] modelPath:onnx-model path
	* \param[in] isCuda:if true,use Ort-GPU,else run it on cpu.
	* \param[in] cudaID:if isCuda==true,run Ort-GPU on cudaID.
	* \param[in] warmUp:if isCuda==true,warm up GPU-model.
	*/
	bool ReadModel(const std::string& modelPath, bool isCuda = false, int cudaID = 0, bool warmUp = true) override;

	/** \brief  detect.
	* \param[in] srcImg:a 3-channels image.
	* \param[out] output:detection results of input image.
	*/
	bool OnnxDetect(cv::Mat& srcImg, std::vector<OutputParams>& output) override;

	/** \brief  detect with GPU preprocessing (zero CPU pixel touching).
	* \param[in] srcImg:a 3-channels image (BGR, uint8).
	* \param[out] output:detection results of input image.
	*/
	bool OnnxDetectGpu(cv::Mat& srcImg, std::vector<OutputParams>& output) override;

	/** \brief  detect,batch size= _batchSize
	* \param[in] srcImg:A batch of images.
	* \param[out] output:detection results of input images.
	*/
	bool OnnxBatchDetect(std::vector<cv::Mat>& srcImg, std::vector<std::vector<OutputParams>>& output) override;

private:

	template <typename T>
	T VectorProduct(const std::vector<T>& v)
	{
		return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
	};
	int PreProcessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params);

	const int _netWidth = 640;   //ONNX-net-input-width
	const int _netHeight = 640;  //ONNX-net-input-height

	int _batchSize = 1;  //if multi-batch,set this
	bool _isDynamicShape = false;//onnx support dynamic shape
	float _classThreshold = 0.5;
	float _nmsThreshold = 0.45;
	float _maskThreshold = 0.5;


	//ONNXRUNTIME	
	Ort::Env _OrtEnv = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Yolov8");
	Ort::SessionOptions _OrtSessionOptions = Ort::SessionOptions();
	Ort::Session* _OrtSession = nullptr;
	Ort::MemoryInfo _OrtMemoryInfo;
	Ort::MemoryInfo _OrtMemoryInfoCuda;

	// GPU Preprocessing context
	GpuPreprocessContext _gpuCtx;
	bool _gpuPreprocessInited = false;
#if ORT_API_VERSION < ORT_OLD_VISON

	char* _inputName, * _output_name0, * _output_name1;
#else
	std::shared_ptr<char> _inputName, _output_name0,_output_name1;
#endif

	std::vector<char*> _inputNodeNames; 
	std::vector<char*> _outputNodeNames;

	size_t _inputNodesNum = 0;        
	size_t _outputNodesNum = 0;       

	ONNXTensorElementDataType _inputNodeDataType; 
	ONNXTensorElementDataType _outputNodeDataType;
	std::vector<int64_t> _inputTensorShape;
	std::vector<int64_t> _outputTensorShape;
	std::vector<int64_t> _outputMaskTensorShape;
};