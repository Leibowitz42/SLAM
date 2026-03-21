#!/bin/bash
# 1. 后台悄悄启动硬件监控，并把输出存到文件
sudo tegrastats --interval 500 --logfile slam_load.txt &
TEGRA_PID=$! # 记住监控程序的进程号

# 2. 跑你的 SLAM 程序 (下面这句换成你运行 SLAM 的实际命令)
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /home/waitangwen/Datasets/rgbd_dataset_freiburg3_walking_xyz Examples/RGB-D/fr3_walking_xyz.txt

# 3. SLAM 运行完毕后，下面这句会自动执行：杀死刚刚后台启动的 tegrastats 进程
sudo kill $TEGRA_PID
echo "测试完毕，数据已保存到 slam_load.txt 中"
