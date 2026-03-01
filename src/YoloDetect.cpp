#include <iostream>
#include "yolov8_seg_onnx.h"
#include "yolov8_seg_trt.h"
#include <opencv2/core.hpp> // For cv::Scalar
#include<time.h>
#include<opencv2/opencv.hpp>
#include <YoloDetect.h>
#include <set>

static bool endsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

YoloDetection::YoloDetection()
{
    std::cout << "Loading Yolo model..." << std::endl;

    std::string model_path_seg = "models/yolo26n-seg.onnx";

    if (endsWith(model_path_seg, ".engine"))
        mpModel = new Yolov8SegTrt();
    else
        mpModel = new Yolov8SegOnnx();

    if (mpModel->ReadModel(model_path_seg, true)) {
        std::cout << "read net ok!" << std::endl;
    } else {
        std::cout << "read net failed!" << std::endl;
    }

    mvDynamicNames = {"person"};
    mvCandidateNames = {"chair", "book"};
    
    // 初始化实例跟踪器
    mNextGlobalID = 1;  // 全局ID从1开始

}

YoloDetection::~YoloDetection()
{
    mvDynamicNames.clear();
    mvCandidateNames.clear();
    mvDynamicArea.clear();
    mvPersonArea.clear();
    mvDynamicMask.clear();
    mvCandidateMask.clear();
    mmDetectMap.clear();
    
    mTrackedInstances.clear();
    mAvailableIDs.clear();
    
    mask.release();
    objectMask.release();
    mInstanceMap.release();
    
    if (mpModel) {
        delete mpModel;
        mpModel = nullptr;
    }
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
    if (mpModel->OnnxDetect(image, result)) {
        
        // ========== 步骤1: 收集半动态物体的bbox和类别，用于实例跟踪 ==========
        std::vector<cv::Rect2f> candidateBboxes;
        std::vector<std::string> candidateClassNames;
        std::vector<int> candidateIndices;  // 记录在result中的索引
        
        const int numClasses = static_cast<int>(mpModel->_className.size());
        for (int i = 0; i < result.size(); i++) {
            int cid = result[i].id;
            if (cid < 0 || cid >= numClasses) continue;
            if (count(mvCandidateNames.begin(), mvCandidateNames.end(), 
                     mpModel->_className[cid])) {
                candidateBboxes.push_back(result[i].box);
                candidateClassNames.push_back(mpModel->_className[cid]);
                candidateIndices.push_back(i);
            }
        }
        
        // ========== 步骤2: 更新实例跟踪器，获取稳定的全局ID ==========
        std::vector<int> globalIDs = UpdateInstanceTracker(candidateBboxes, candidateClassNames);
        
        // ========== 步骤3: 生成mInstanceMap，使用稳定的全局ID ==========
        mask = cv::Mat::zeros(image.size(), CV_8UC3);
        mInstanceMap.setTo(0); // 彻底清空
        
        const int numColors = static_cast<int>(color.size());
        for (int i = 0; i < result.size(); i++) {
            int cid = result[i].id;
            if (cid < 0 || cid >= numClasses) continue;
            if (cid >= numColors) continue;

            std::string className = mpModel->_className[cid];
            // Only process if the class is in our dynamic or candidate list
            if (count(mvDynamicNames.begin(), mvDynamicNames.end(), className) == 0 &&
                count(mvCandidateNames.begin(), mvCandidateNames.end(), className) == 0) {
                continue;
            }

            int left = 0, top = 0;
            if (result[i].box.area() > 0) {
                rectangle(img, result[i].box, color[cid], 2, 8);
                left = result[i].box.x;
                top = result[i].box.y;
            }

            // 1. 可视化部分（保持原样，用于画出带颜色的分割图）
            if (result[i].rotatedBox.size.width * result[i].rotatedBox.size.height > 0) {
                DrawRotatedBox(img, result[i].rotatedBox, color[cid], 2);
                left = result[i].rotatedBox.center.x;
                top = result[i].rotatedBox.center.y;
            }
            
            // add masked image to mvDynamicMask  2. 核心分类逻辑
            if (result[i].boxMask.rows && result[i].boxMask.cols > 0){
                cv::Rect box = result[i].box;
                if (box.x >= 0 && box.y >= 0 && box.x + box.width <= mask.cols && box.y + box.height <= mask.rows)
                    mask(box).setTo(color[cid], result[i].boxMask);
            }
            if (result[i].box.width <= 0 || result[i].box.height <= 0 || result[i].boxMask.empty())
                continue;
                
            cv::Rect box = result[i].box;
            if (box.x < 0 || box.y < 0 || box.x + box.width > mInstanceMap.cols || box.y + box.height > mInstanceMap.rows)
                continue;
            // 检查当前物体的类别名称是否在"预设动态物体列表(mvDynamicNames)"中
            if (count(mvDynamicNames.begin(), mvDynamicNames.end(), className)){
                mInstanceMap(box).setTo(cv::Scalar(255), result[i].boxMask);
            }
            else if (count(mvCandidateNames.begin(), mvCandidateNames.end(), className)){
                // ✅ 使用稳定的全局ID，而不是临时的candidateID
                auto it = std::find(candidateIndices.begin(), candidateIndices.end(), i);
                if (it != candidateIndices.end()) {
                    int idx = std::distance(candidateIndices.begin(), it);
                    if (idx >= 0 && idx < (int)globalIDs.size()) {
                        int stableID = globalIDs[idx];
                        if (stableID > 0 && stableID < 250) {
                            mInstanceMap(box).setTo(cv::Scalar(stableID), result[i].boxMask);
                        }
                    }
                }
            }
            
            // 3. 统计映射（保持原样，供 Viewer 使用）
            cv::Rect2i DetectArea(left, top, result[i].box.width, result[i].box.height);
            mmDetectMap[className].push_back(DetectArea);
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

// ========== 实例跟踪器实现 ==========

// 分配新ID（优先使用回收的ID）
int YoloDetection::AllocateInstanceID() {
    // 1. 优先使用回收的ID
    if (!mAvailableIDs.empty()) {
        int id = *mAvailableIDs.begin();
        mAvailableIDs.erase(mAvailableIDs.begin());
        return id;
    }
    
    // 2. 如果没有回收的ID，使用新ID
    if (mNextGlobalID < MAX_INSTANCE_ID) {
        return mNextGlobalID++;
    }
    
    // 3. 如果ID用完了，从1开始循环使用
    // 这种情况很少发生，因为会有ID回收
    std::cout << "[Instance Tracker] WARNING: ID pool exhausted, recycling from 1" << std::endl;
    mNextGlobalID = 1;
    return mNextGlobalID++;
}

// 回收ID（当实例被删除时）
void YoloDetection::RecycleInstanceID(int id) {
    if (id > 0 && id < MAX_INSTANCE_ID) {
        mAvailableIDs.insert(id);
        // std::cout << "[Instance Tracker] Recycled ID: " << id << std::endl;
    }
}

float YoloDetection::ComputeIoU(const cv::Rect2f& box1, const cv::Rect2f& box2) {
    // 计算交集区域
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    
    // 计算并集区域
    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    float union_area = area1 + area2 - intersection;
    
    // 返回IoU
    return intersection / (union_area + 1e-6f);
}

std::vector<int> YoloDetection::UpdateInstanceTracker(
    const std::vector<cv::Rect2f>& bboxes, 
    const std::vector<std::string>& classNames) {
    
    const float IOU_THRESHOLD = 0.3f;  // IoU阈值：大于此值认为是同一物体
    const int MAX_FRAMES_LOST = 30;    // 最大丢失帧数：超过此值删除实例
    
    std::vector<int> detectionToGlobalID(bboxes.size(), -1);
    std::vector<bool> trackedMatched(mTrackedInstances.size(), false);
    std::vector<TrackedInstance> newInstances;  // 收集新实例
    
    // 1. 对每个新检测，找到最佳匹配的已跟踪实例
    for (size_t i = 0; i < bboxes.size(); i++) {
        const cv::Rect2f& newBox = bboxes[i];
        const std::string& className = classNames[i];
        
        float bestIoU = IOU_THRESHOLD;
        int bestMatchIdx = -1;
        
        // 在已跟踪实例中寻找最佳匹配
        for (size_t j = 0; j < mTrackedInstances.size(); j++) {
            if (trackedMatched[j]) continue;  // 已被匹配
            if (mTrackedInstances[j].className != className) continue;  // 类别不同
            
            float iou = ComputeIoU(newBox, mTrackedInstances[j].bbox);
            if (iou > bestIoU) {
                bestIoU = iou;
                bestMatchIdx = j;
            }
        }
        
        if (bestMatchIdx >= 0) {
            // 找到匹配：更新已有实例
            trackedMatched[bestMatchIdx] = true;
            mTrackedInstances[bestMatchIdx].bbox = newBox;
            mTrackedInstances[bestMatchIdx].framesSinceLastSeen = 0;
            detectionToGlobalID[i] = mTrackedInstances[bestMatchIdx].globalID;
        } else {
            // 未找到匹配：先收集新实例，不立即添加
            TrackedInstance newInstance;
            newInstance.globalID = AllocateInstanceID();  // 使用ID分配函数
            newInstance.bbox = newBox;
            newInstance.className = className;
            newInstance.framesSinceLastSeen = 0;
            
            newInstances.push_back(newInstance);
            detectionToGlobalID[i] = newInstance.globalID;
            
            std::cout << "[Instance Tracker] New instance: ID=" << newInstance.globalID 
                      << ", class=" << className << std::endl;
        }
    }
    
    // 循环结束后，统一添加新实例
    for (const auto& newInst : newInstances) {
        mTrackedInstances.push_back(newInst);
    }
    
    // 2. 更新未匹配的实例，并移除长时间未见到的实例
    auto it = mTrackedInstances.begin();
    size_t idx = 0;
    
    while (it != mTrackedInstances.end()) {
        if (idx < trackedMatched.size() && !trackedMatched[idx]) {
            it->framesSinceLastSeen++;
            
            if (it->framesSinceLastSeen > MAX_FRAMES_LOST) {
                std::cout << "[Instance Tracker] Lost: ID=" << it->globalID 
                          << ", class=" << it->className << std::endl;
                
                // 回收ID
                RecycleInstanceID(it->globalID);
                
                it = mTrackedInstances.erase(it);
                // 注意：erase后不增加idx，因为后面的元素会前移
                continue;
            }
        }
        ++it;
        ++idx;
    }
    
    return detectionToGlobalID;
}
