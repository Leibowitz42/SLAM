#!/bin/bash
set -e

cd "$(dirname "$0")"

# 1. 检查 build 目录是否存在，不存在则创建
if [ ! -d "build" ]; then
    echo ">>> build 目录不存在，正在初始化..."
    mkdir build
fi

cd build

# 2. 只有在没有生成过 build 配置文件时才运行完整的 cmake
# 或者当你修改了 CMakeLists.txt 时，cmake 也会自动被 ninja 触发
if [ ! -f "build.ninja" ]; then
    echo ">>> 运行初次 CMake 配置..."
    cmake -G Ninja ..
fi

# 内存 8GB，限制并行数为2
echo ">>> 开始编译 (限制并行任务为3)..."
ninja -j3

echo ">>> 编译完成！"

