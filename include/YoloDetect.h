/*
* This file is part of YDM-SLAM
* Author: Balveer Singh
* GitHub: https://github.com/balveersinghyt/YDM-SLAM
*/
#ifndef YOLO_DETECT_H
#define YOLO_DETECT_H

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <utility>
#include <time.h>
using namespace std;
#include "yolov8_seg_base.h"

class YoloDetection
{
public:
    YoloDetection();
    ~YoloDetection();
    void GetImage(cv::Mat& RGB);
    void ClearImage();
    bool Detect();
    void ClearArea();
    vector<cv::Rect2i> mvPersonArea = {};

public:
    cv::Mat mRGB;
    std::vector<std::string> mClassnames;

    // 6-28
    vector<string> mvDynamicNames;
    vector<string> mvCandidateNames;
    
    vector<cv::Rect2i> mvDynamicArea;
    vector<cv:: Mat> mvDynamicMask;
    vector<cv::Mat> mvCandidateMask;

    // map for detection and mask together
    map<string, vector<cv::Rect2i>> mmDetectMap;
    cv::Mat mask;
    cv::Mat objectMask;
    cv::Mat mInstanceMap;
    // Model pointer: TensorRT (.engine) or ONNX Runtime (.onnx)
    IYolov8Seg* mpModel = nullptr;

private:
    // ========== 实例跟踪器 ==========
    // 用于为每个物体实例分配稳定的全局ID，解决多实例独立判定问题
    
    struct TrackedInstance {
        int globalID;               // 全局唯一ID（跨帧稳定）
        cv::Rect2f bbox;            // 边界框
        std::string className;      // 类别名称
        int framesSinceLastSeen;    // 未见到的帧数（用于处理遮挡）
        
        TrackedInstance() : globalID(-1), framesSinceLastSeen(0) {}
    };
    
    std::vector<TrackedInstance> mTrackedInstances;  // 已跟踪的实例列表
    int mNextGlobalID;  // 下一个可用的全局ID
    
    // ========== ID回收机制 ==========
    // 限制ID范围在1-250之间（255保留给绝对动态物体）
    static const int MAX_INSTANCE_ID = 250;
    std::set<int> mAvailableIDs;  // 可用的ID池（已回收的ID）
    
    // 分配新ID（优先使用回收的ID）
    int AllocateInstanceID();
    
    // 回收ID（当实例被删除时）
    void RecycleInstanceID(int id);
    
    // 计算两个边界框的IoU（交并比）
    float ComputeIoU(const cv::Rect2f& box1, const cv::Rect2f& box2);
    
    // 更新实例跟踪器，返回每个检测的稳定全局ID
    std::vector<int> UpdateInstanceTracker(const std::vector<cv::Rect2f>& bboxes, 
                                          const std::vector<std::string>& classNames);

};


#endif //YOLO_DETECT_H
