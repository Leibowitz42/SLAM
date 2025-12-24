#include <iostream>
#include "yolov8_seg_onnx.h"
#include <opencv2/core.hpp> // For cv::Scalar
#include<time.h>
#include<opencv2/opencv.hpp>
#include <YoloDetect.h>

// using namespace dnn;
Yolov8SegOnnx		model;
YoloDetection::YoloDetection()
{
    std::cout << "Loading Yolo model..." << std::endl;

    std::string model_path_seg = "models/yolo11s-seg.onnx";

	// loading model
    if (model.ReadModel(model_path_seg, true)) {
		std:: cout << "read net ok!" << endl;
	}
    else {
        std:: cout << "read net failed!" << endl;
		// return -1;
	}

    mvDynamicNames = {"person", "car", "motorbike", "bus", "train", "truck", "boat", "bird", "cat",
                      "dog", "horse", "sheep", "crow", "bear"};
    mvCandidateNames = {"chair", "book"};

}

YoloDetection::~YoloDetection()
{

}

bool YoloDetection::Detect()
{

    // loading image
    cv::Mat img;
    if(mRGB.empty())
    {
        std::cout << "Read image failed!" << std::endl;
        return -1;
    }
    cv:: cvtColor(mRGB, img, cv::COLOR_BGR2RGB);
    cv:: Mat image = img.clone();

    std::vector<cv::Scalar> color;
    srand(time(0));
    for (int i = 0; i < 80; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(cv::Scalar(b, g, r));
    }
    std::vector<OutputParams> result;
    mInstanceMap = cv::Mat::zeros(image.size(), CV_8UC1);
    objectMask = cv::Mat::zeros(image.size(), CV_8UC1);
    // std:: cout<< "objectMask size: " << objectMask.size() << " image size: " << image.size() << endl;
    if (model.OnnxDetect(image, result)) {
        mask = cv::Mat::zeros(image.size(), CV_8UC3);
        int candidateID = 1; // 给半动态物体的起始 ID (1-250)
        for (int i = 0; i < result.size(); i++) {
            int left, top;
            int color_num = i;
            if (result[i].box.area() > 0) {
                rectangle(img, result[i].box, color[result[i].id], 2, 8);
                left = result[i].box.x;
                top = result[i].box.y;
            }

            // 1. 可视化部分（保持原样，用于画出带颜色的分割图）
            if (result[i].rotatedBox.size.width * result[i].rotatedBox.size.height > 0) {
                DrawRotatedBox(img, result[i].rotatedBox, color[result[i].id], 2);
                left = result[i].rotatedBox.center.x;
                top = result[i].rotatedBox.center.y;
            }
            
            // add masked image to mvDynamicMask  2. 核心分类逻辑
            if (result[i].boxMask.rows && result[i].boxMask.cols > 0){
                mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
            }
            // 检查当前物体的类别名称是否在“预设动态物体列表(mvDynamicNames)”中
            if (count(mvDynamicNames.begin(), mvDynamicNames.end(), model._className[result[i].id])){
                mInstanceMap(result[i].box).setTo(cv::Scalar(255), result[i].boxMask);
            }
            else if (count(mvCandidateNames.begin(), mvCandidateNames.end(), model._className[result[i].id])){
                // a. 生成黑白掩码：在 objectMask 的物体区域内，将属于物体的像素设为 255 (白色)
                mInstanceMap(result[i].box).setTo(cv::Scalar(candidateID), result[i].boxMask);
                // 增加 ID，确保下一把椅子有不同的编号
                candidateID++; 
                if(candidateID >= 255) candidateID = 254; // 防止溢出
            }
            
            // 3. 统计映射（保持原样，供 Viewer 使用）
            cv:: Rect2i DetectArea(left, top, (result[i].box.width), (result[i].box.height));
            mmDetectMap[model._className[result[i].id]].push_back(DetectArea);
           
        }
        if (mvDynamicArea.size() == 0)
        {
            cv::Rect2i tDynamicArea(1, 1, 1, 1);
            mvDynamicArea.push_back(tDynamicArea);
        }

    }
    else
        cout << "Detect Failed!" << endl;
    
    return true;
}


void YoloDetection::GetImage(cv::Mat &RGB)
{
    mRGB = RGB;
}

void YoloDetection::ClearImage()
{
    mRGB = 0;
}

void YoloDetection::ClearArea()
{
    mvPersonArea.clear();
}
