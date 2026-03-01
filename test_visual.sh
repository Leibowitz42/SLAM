#!/bin/bash
# 用法: ./test_visual.sh [结果目录]
# 例如: ./test_visual.sh /home/mickey/Datasets/rgbd_dataset_freiburg3_walking_xyz/Results

RESULTS_DIR="${1:-$HOME/Datasets/rgbd_dataset_freiburg3_walking_xyz/Results}"
if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "错误: 结果目录不存在: $RESULTS_DIR"
    echo "请先运行 run.sh 生成轨迹，或指定正确路径: ./test_visual.sh /path/to/Results"
    exit 1
fi
cd "$RESULTS_DIR" || exit 1

if ! command -v evo_ape &>/dev/null; then
    echo "未找到 evo 工具。请安装: pip install evo --upgrade"
    exit 1
fi

echo "=========================================="
echo "生成轨迹评估可视化"
echo "=========================================="

echo ""
echo "1. 评估关键帧轨迹 (APE)..."
evo_ape tum groundtruth.txt KeyFrameTrajectory.txt \
    -va \
    --plot \
    --plot_mode xy \
    --save_plot kf_ape.png \
    --save_results kf_ape.zip

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
