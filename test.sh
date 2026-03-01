#!/bin/bash
# 用法: ./test.sh [结果目录]
# 例如: ./test.sh /home/mickey/Datasets/rgbd_dataset_freiburg3_walking_xyz/Results
# 程序会把轨迹保存到 数据集路径/Results/，运行前先跑 run.sh 生成轨迹

RESULTS_DIR="${1:-$HOME/Datasets/rgbd_dataset_freiburg3_walking_xyz/Results}"
if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "错误: 结果目录不存在: $RESULTS_DIR"
    echo "请先运行 run.sh 生成轨迹，或指定正确路径: ./test.sh /path/to/Results"
    exit 1
fi
cd "$RESULTS_DIR" || exit 1

if ! command -v evo_ape &>/dev/null || ! command -v evo_rpe &>/dev/null; then
    echo "未找到 evo 工具。请安装: pip install evo --upgrade"
    exit 1
fi

evo_ape tum groundtruth.txt KeyFrameTrajectory.txt -v -a --plot
evo_rpe tum groundtruth.txt CameraTrajectory.txt -va --delta 0.5 --delta_unit m --plot
