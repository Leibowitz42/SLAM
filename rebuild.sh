#!/bin/bash
set -e

cd "$(dirname "$0")"

echo ">>> 删除旧的 build 目录..."
rm -rf build

echo ">>> 创建新的 build 目录..."
mkdir build
cd build

echo ">>> 运行 CMake..."
cmake .. 
# 内存 8GB，限制并行数为2
echo ">>> 开始编译 (限制并行任务为2)..."
make -j3

echo ">>> 编译完成！"

