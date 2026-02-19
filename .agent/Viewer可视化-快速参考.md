# Viewer可视化增强 - 快速参考

## 🎨 新增功能总览

### 1. 彩色实例标记
- **绿色**：静态背景
- **彩色**：不同的椅子/书本实例（每个ID一个颜色）
- **红色**：动态物体

### 2. 实例ID显示
- 在特征点旁边显示数字ID
- 只对半动态物体显示（ID 1-254）

### 3. 统计信息
- 顶部状态栏显示：`Instances: N`
- N = 当前帧中的实例数量

---

## 🚀 快速使用

```bash
# 编译
cd /home/waitangwen/SLAM
./rebuild.sh 1

# 运行
./run.sh
```

**观察Viewer窗口**：
- 不同颜色 = 不同实例
- 红色 = 动态物体
- 数字 = 实例ID

---

## 📊 颜色示例

| 实例 | ID | 颜色 | 状态 |
|------|----|----|------|
| 椅子1 | 1 | 🟡 黄绿 | 静态 |
| 椅子2 | 2 | 🟢 绿色 | 静态 |
| 椅子3 | 16 | 🔴 红色 | 动态 |
| 书本 | 33 | 🟡 黄色 | 静态 |
| 背景 | -1 | 🟢 绿色 | 静态 |

---

## 🔧 修改的文件

1. **include/MapPoint.h**
   - 添加 `GetInstanceID()` 方法

2. **include/FrameDrawer.h**
   - 添加 `GetInstanceColor()` 方法声明

3. **src/FrameDrawer.cc**
   - 实现颜色生成算法
   - 修改特征点绘制逻辑
   - 添加实例ID文本显示
   - 添加实例统计信息

---

## 📝 技术细节

### 颜色生成

```cpp
// 使用HSV色彩空间
int hue = (instanceID * 37) % 180;  // 质数确保均匀分布
cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
```

### ID显示

```cpp
// 在特征点右上方显示ID
std::string idText = std::to_string(instanceID);
cv::putText(im, idText, 
           cv::Point(point.x + 8, point.y - 8),
           cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
```

---

## ✅ 预期效果

运行后，在Viewer窗口中应该看到：

1. **不同颜色的特征点**
   - 每个椅子有独特的颜色
   - 同一椅子在连续帧中保持相同颜色

2. **实例ID数字**
   - 在每个半动态物体的特征点旁边
   - 数字稳定，不频繁变化

3. **动态物体标记**
   - 移动的椅子变为红色
   - 红色特征点逐渐消失（被剔除）

4. **统计信息**
   - 顶部显示实例数量
   - 数字稳定在2-5之间

---

## 🐛 故障排除

### 问题1：看不到颜色
**原因**：实例ID未正确分配  
**检查**：
```bash
./run.sh 2>&1 | grep "Instance Tracker"
```
应该看到 "New instance: ID=X"

### 问题2：颜色频繁变化
**原因**：IoU匹配不稳定  
**解决**：调整 `IOU_THRESHOLD`（YoloDetect.cpp）

### 问题3：看不到ID文本
**原因**：文本太小或被遮挡  
**解决**：调整字体大小（FrameDrawer.cc line 250）

---

**创建时间**：2026-02-16 12:10  
**状态**：✅ 实现完成，编译中  
**文档**：详见 `Viewer可视化增强.md`
