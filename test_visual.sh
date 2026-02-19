#!/bin/bash

# 进入结果目录
cd /home/waitangwen/Datasets/rgbd_dataset_freiburg3_walking_xyz/Results

echo "=========================================="
echo "生成轨迹评估可视化"
echo "=========================================="

# 1. 关键帧轨迹的绝对位姿误差 (APE)
echo ""
echo "1. 评估关键帧轨迹 (APE)..."
evo_ape tum groundtruth.txt KeyFrameTrajectory.txt \
    -va \
    --plot \
    --plot_mode xy \
    --save_plot kf_ape.png \
    --save_results kf_ape.zip

# 2. 相机轨迹的相对位姿误差 (RPE)
echo ""
echo "2. 评估相机轨迹 (RPE)..."
evo_rpe tum groundtruth.txt CameraTrajectory.txt \
    -va \
    --delta 0.5 \
    --delta_unit m \
    --plot \
    --plot_mode xy \
    --save_plot cam_rpe.png \
    --save_results cam_rpe.zip

# 3. 完整轨迹对比
echo ""
echo "3. 生成轨迹对比图..."
evo_traj tum groundtruth.txt \
    --ref KeyFrameTrajectory.txt \
    CameraTrajectory.txt \
    -va \
    --plot \
    --plot_mode xy \
    --save_plot trajectory_comparison.png

echo ""
echo "=========================================="
echo "可视化完成！生成的文件："
echo "  - kf_ape.png: 关键帧绝对位姿误差"
echo "  - cam_rpe.png: 相机相对位姿误差"
echo "  - trajectory_comparison.png: 轨迹对比"
echo "=========================================="
