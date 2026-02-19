# Viewer可视化增强 - 实例ID显示

## 🎨 新增功能

### 1. 彩色实例标记

每个实例使用不同的颜色显示，方便区分：

**颜色方案**：
- **绿色**：静态背景点（instanceID = -1 或 0）
- **彩色**：半动态物体实例（instanceID = 1-254）
  - 使用HSV色彩空间生成
  - 每个ID对应一个独特的颜色
  - 高饱和度、高亮度，视觉效果好
- **红色**：动态物体（instanceID = 255 或被标记为动态）

### 2. 实例ID文本显示

在每个半动态物体的特征点旁边显示其实例ID：
- 文本位置：特征点右上方（偏移8像素）
- 字体：HERSHEY_SIMPLEX，大小0.4
- 颜色：与特征点颜色一致

### 3. 实例统计信息

在Viewer顶部状态栏显示当前帧中的实例数量：
```
SLAM MODE | Maps: 1, KFs: 66, MPs: 12345, Matches: 1500 | Instances: 3
```

---

## 🔧 实现细节

### 颜色生成算法

```cpp
cv::Scalar GetInstanceColor(int instanceID, bool isDynamic)
{
    if (isDynamic) {
        return cv::Scalar(0, 0, 255);  // 红色
    }
    
    if (instanceID <= 0 || instanceID >= 255) {
        return cv::Scalar(0, 255, 0);  // 绿色
    }
    
    // HSV色彩空间
    int hue = (instanceID * 37) % 180;  // 37是质数，确保均匀分布
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    
    return cv::Scalar(bgr.at<cv::Vec3b>(0, 0));
}
```

**为什么使用37？**
- 37是质数，与180互质
- 确保连续的ID生成差异明显的颜色
- 避免相邻ID颜色过于相似

### 绘制流程

```cpp
// 1. 获取MapPoint的实例ID
MapPoint* pMP = currentFrame.mvpMapPoints[i];
int instanceID = pMP->GetInstanceID();

// 2. 从当前帧的InstanceMap验证
uchar id = currentFrame.mInstanceMap.at<uchar>(v, u);
if (id > 0 && id < 255) {
    instanceID = id;  // 使用最新的ID
}

// 3. 根据ID选择颜色
cv::Scalar color = GetInstanceColor(instanceID, isDynamic);

// 4. 绘制特征点
cv::rectangle(im, pt1, pt2, color);
cv::circle(im, point, 2, color, -1);

// 5. 显示ID文本
if (instanceID > 0 && instanceID < 255) {
    std::string idText = std::to_string(instanceID);
    cv::putText(im, idText, 
               cv::Point(point.x + 8, point.y - 8),
               cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
}
```

---

## 📊 可视化效果

### 示例场景

假设当前帧有：
- 椅子1（ID=1）：蓝色
- 椅子2（ID=2）：青色
- 椅子3（ID=16，动态）：红色
- 书本（ID=33）：黄色
- 静态背景：绿色

**Viewer显示**：
```
┌─────────────────────────────────────────┐
│ SLAM MODE | ... | Instances: 4          │
├─────────────────────────────────────────┤
│                                         │
│    ●1 (蓝色)    ●2 (青色)              │
│                                         │
│         ●16 (红色)                      │
│                                         │
│    ●33 (黄色)                           │
│                                         │
│  ● ● ● (绿色，静态背景)                │
│                                         │
└─────────────────────────────────────────┘
```

### 颜色示例

| Instance ID | Hue | 颜色 | 说明 |
|-------------|-----|------|------|
| 1 | 37 | 🟡 黄绿 | 椅子1 |
| 2 | 74 | 🟢 绿色 | 椅子2 |
| 3 | 111 | 🔵 蓝色 | 椅子3 |
| 16 | - | 🔴 红色 | 动态椅子 |
| 32 | 4 | 🔴 红色系 | 椅子32 |
| 33 | 41 | 🟡 黄色 | 书本 |
| -1/0 | - | 🟢 绿色 | 静态背景 |

---

## 🎯 使用方法

### 运行SLAM

```bash
cd /home/waitangwen/SLAM
./rebuild.sh
./run.sh
```

### 观察可视化

1. **打开Viewer窗口**
   - 窗口标题："ORB-SLAM3: Current Frame"
   - 显示当前帧的图像和特征点

2. **查看实例颜色**
   - 不同颜色的特征点代表不同实例
   - 红色特征点表示动态物体
   - 绿色特征点表示静态背景

3. **查看实例ID**
   - 每个半动态物体的特征点旁边显示数字
   - 数字就是实例的全局ID

4. **查看统计信息**
   - 窗口顶部显示"Instances: N"
   - N是当前帧中的实例数量

---

## 🔍 调试技巧

### 验证实例跟踪

通过观察颜色变化，可以验证实例跟踪是否稳定：

**正常情况**：
- 同一椅子在连续帧中保持相同颜色
- ID数字不变

**异常情况**：
- 颜色频繁变化 → IoU匹配失败
- ID频繁变化 → 跟踪不稳定

### 验证动态检测

观察颜色变化：

**静态椅子**：
- 保持彩色（非红色）
- ID稳定

**动态椅子**：
- 变为红色
- 特征点被剔除（消失）

### 验证实例数量

观察"Instances"数字：

**正常情况**：
- 数字稳定在2-5之间（取决于场景）
- 缓慢变化

**异常情况**：
- 数字频繁跳变 → 检测不稳定
- 数字持续增长 → ID未正确回收

---

## 📝 修改的文件

### 1. include/FrameDrawer.h

添加：
```cpp
// 根据实例ID生成颜色（用于可视化不同实例）
cv::Scalar GetInstanceColor(int instanceID, bool isDynamic);
```

### 2. src/FrameDrawer.cc

**添加的功能**：
1. `GetInstanceColor()` 函数实现
2. 在`DrawFrame()`中根据实例ID选择颜色
3. 在`DrawFrame()`中显示实例ID文本
4. 在`DrawTextInfo()`中显示实例统计

**添加的头文件**：
```cpp
#include<opencv2/imgproc/imgproc.hpp>  // cvtColor
#include<set>                           // std::set
#include<string>                        // std::to_string
```

---

## 🎨 颜色分布示例

使用质数37确保颜色均匀分布：

```
ID   Hue  Color
1    37   黄绿
2    74   绿
3    111  蓝
4    148  紫
5    5    红
6    42   黄
7    79   绿
8    116  青
9    153  蓝紫
10   10   红橙
...
```

每个ID都有独特的颜色，视觉上易于区分。

---

## 🚀 未来改进

### 1. 颜色方案优化

- [ ] 使用更鲜艳的颜色
- [ ] 避免与背景颜色冲突
- [ ] 支持用户自定义配色

### 2. 显示优化

- [ ] 可选择是否显示ID文本
- [ ] 调整文本大小和位置
- [ ] 添加半透明背景提高可读性

### 3. 交互功能

- [ ] 点击特征点显示详细信息
- [ ] 过滤显示特定实例
- [ ] 高亮显示动态实例

### 4. 统计信息增强

- [ ] 显示每个实例的点数
- [ ] 显示动态实例数量
- [ ] 显示实例的类别名称

---

## 📚 参考资料

### HSV色彩空间

- **H (Hue)**：色调，0-180度
- **S (Saturation)**：饱和度，0-255
- **V (Value)**：明度，0-255

### OpenCV绘制函数

- `cv::circle()`：绘制圆形
- `cv::rectangle()`：绘制矩形
- `cv::putText()`：绘制文本
- `cv::cvtColor()`：颜色空间转换

---

**创建时间**：2026-02-16 12:00  
**状态**：✅ 实现完成，编译中  
**作者**：Antigravity AI Assistant
