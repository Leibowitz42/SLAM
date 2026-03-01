#!/bin/bash
# 在能复现段错误的环境下运行此脚本，崩溃时会打印并保存 backtrace
# 用法: ./run_gdb_segfault.sh [数据集目录]
# 例如: ./run_gdb_segfault.sh /path/to/rgbd_dataset_freiburg3_walking_xyz

set -e
cd "$(dirname "$0")"
DATASET="${1:-/home/waitangwen/Datasets/rgbd_dataset_freiburg3_walking_xyz}"
BT_FILE="gdb_backtrace.txt"

echo "Dataset: $DATASET"
echo "On segfault, backtrace will be written to $BT_FILE"
echo "---"

gdb -batch \
  -ex "set pagination off" \
  -ex "run" \
  -ex "bt full" \
  -ex "info registers" \
  -ex "quit" \
  --args ./Examples/RGB-D/rgbd_tum \
    Vocabulary/ORBvoc.txt \
    Examples/RGB-D/TUM3.yaml \
    "$DATASET" \
    Examples/RGB-D/fr3_walking_xyz.txt 2>&1 | tee "$BT_FILE"

echo "---"
echo "Output saved to $BT_FILE"
