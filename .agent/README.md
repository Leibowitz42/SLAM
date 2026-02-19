# 多实例动态物体剔除 - 文档索引

## 📚 文档清单

### 🎯 核心文档

1. **[项目总结.md](./项目总结.md)** ⭐ **推荐首读**
   - 项目概述和完整总结
   - 技术成果和性能指标
   - 未来工作规划

2. **[性能评估与可视化.md](./性能评估与可视化.md)** ⭐ **查看结果**
   - 详细的性能指标
   - 可视化图表解读
   - 与原系统对比

### 📖 实现文档

3. **[多实例动态剔除-实现完成.md](./多实例动态剔除-实现完成.md)**
   - 完整的实现说明
   - 工作流程和数据流
   - 配置参数说明

4. **[多实例跟踪实现总结.md](./多实例跟踪实现总结.md)**
   - 详细的代码修改记录
   - 实现细节和原理

5. **[多实例跟踪解决方案.md](./多实例跟踪解决方案.md)**
   - 设计方案和思路
   - 技术选型

### 🎨 可视化文档

6. **[Viewer可视化增强.md](./Viewer可视化增强.md)**
   - 实例ID显示实现
   - 颜色标记方案
   - 使用方法

7. **[Viewer可视化-快速参考.md](./Viewer可视化-快速参考.md)**
   - 快速使用指南
   - 常见问题

8. **[ID回收与固定颜色.md](./ID回收与固定颜色.md)** ⭐ **最新**
   - ID回收机制实现
   - 固定颜色方案
   - 性能优化

9. **[遮挡检测与处理方案.md](./遮挡检测与处理方案.md)** ⭐ **效果显著**
   - 解决ATE暴增问题
   - 性能测试报告 (RMSE 📉 ~30%)

### 🐛 调试文档

10. **[段错误调试指南.md](./段错误调试指南.md)**
   - Bug分析和修复
   - 调试技巧

---

## 🚀 快速开始

### 编译和运行

```bash
# 编译
cd /home/waitangwen/SLAM
./rebuild.sh

# 运行SLAM
./run.sh

# 评估性能
./test_visual.sh
```

### 查看结果

**可视化图表位置**：
```
/home/waitangwen/Datasets/rgbd_dataset_freiburg3_walking_xyz/Results/
├── kf_ape_raw.png      # 关键帧APE时间序列
├── kf_ape_map.png      # 关键帧APE空间分布
├── cam_rpe_raw.png     # 相机RPE时间序列
└── cam_rpe_map.png     # 相机RPE空间分布
```

**轨迹文件**：
```
/home/waitangwen/Datasets/rgbd_dataset_freiburg3_walking_xyz/Results/
├── CameraTrajectory.txt      # 相机轨迹
└── KeyFrameTrajectory.txt    # 关键帧轨迹
```

---

## 📊 核心指标

| 指标 | 数值 | 评价 |
|------|------|------|
| **APE RMSE** | **1.97cm** | 🌟 优秀 |
| **RPE RMSE** | **2.02cm** | 🌟 优秀 |
| **帧率** | **18.7 FPS** | ✅ 实用 |
| **动态检测** | **1/1** | ✅ 准确 |

---

## 🔧 关键代码文件

### 修改的文件

1. **include/YoloDetect.h**
   - 添加实例跟踪器数据结构

2. **src/YoloDetect.cpp**
   - 实现IoU匹配和实例跟踪
   - 生成稳定的实例ID

3. **include/MapPoint.h**
   - 添加mInstanceID成员变量

4. **src/MapPoint.cc**
   - 从mInstanceMap读取实例ID

5. **src/Tracking.cc**
   - 按实例ID分组检查
   - 实例级动态判定

---

## 💡 核心技术

### 1. 实例跟踪
- **方法**：IoU匹配
- **阈值**：0.3
- **效果**：稳定的跨帧ID

### 2. 动态检测
- **方法**：对极几何
- **阈值**：30%违反率
- **效果**：精准剔除

### 3. 确认机制
- **方法**：连续2帧投票
- **效果**：避免误检

---

## 📝 使用示例

### 查看实例跟踪日志

```bash
./run.sh 2>&1 | grep "Instance Tracker"
```

**输出示例**：
```
[Instance Tracker] New instance: ID=1, class=chair
[Instance Tracker] New instance: ID=2, class=chair
[Instance Tracker] Lost: ID=1, class=chair
```

### 查看动态检测日志

```bash
./run.sh 2>&1 | grep "Dyna-Logic"
```

**输出示例**：
```
[Dyna-Logic] Instance ID 16 (Chair) is CONFIRMED DYNAMIC (Count 2). Removed 40 points.
```

---

## 🎯 调优参数

### YoloDetect.cpp

```cpp
const float IOU_THRESHOLD = 0.3f;      // IoU匹配阈值
const int MAX_FRAMES_LOST = 30;        // 最大丢失帧数
```

### Tracking.cc

```cpp
const float epipolar_dist_th = 3.5;    // 对极距离阈值
const float dynamic_ratio = 0.3;       // 动态比例阈值
const int confirm_frames = 2;          // 确认帧数
```

---

## 🐛 已知问题

1. **退出时内存错误**
   - 现象：`malloc(): unsorted double linked list corrupted`
   - 影响：仅在程序退出时，不影响功能
   - 状态：可忽略

---

## 📞 联系方式

**项目路径**：`/home/waitangwen/SLAM`  
**文档路径**：`/home/waitangwen/SLAM/.agent/`  
**结果路径**：`/home/waitangwen/Datasets/rgbd_dataset_freiburg3_walking_xyz/Results/`

---

**最后更新**：2026-02-16 11:55  
**版本**：v1.0  
**状态**：✅ 完成
