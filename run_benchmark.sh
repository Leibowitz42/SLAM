#!/bin/bash

# 强制使用非交互式后端绘图，防止卡死
export MPLBACKEND=Agg


# 配置
# DATASET_PATH="/home/waitangwen/Datasets/rgbd_dataset_freiburg3_walking_xyz"
DATASET_PATH="/home/waitangwen/Datasets/rgbd_dataset_freiburg3_sitting_static"
RESULT_DIR="$DATASET_PATH/Results"
GROUNDTRUTH="$DATASET_PATH/groundtruth.txt"
RUN_COUNT=5

# 创建结果目录（如果不存在）
mkdir -p $RESULT_DIR

echo "=========================================="
echo "🚀 开始 SLAM 性能基准测试 (运行 $RUN_COUNT 次)"
echo "=========================================="
echo "评估标准:"
echo "  1. ATE (绝对轨迹误差): 针对 KeyFrameTrajectory (全局一致性)"
echo "  2. RPE Trans (相对轨迹误差-平移): 针对 CameraTrajectory (局部漂移)"
echo "  3. RPE Rot (相对轨迹误差-旋转): 针对 CameraTrajectory"
echo "=========================================="

# 清理之前的临时文件
rm -f $RESULT_DIR/CameraTrajectory_*.txt
rm -f $RESULT_DIR/KeyFrameTrajectory_*.txt

total_ate_rmse=0
total_rpe_rmse=0
total_rpe_rot_rmse=0
count_success=0

for i in $(seq 1 $RUN_COUNT); do
    echo ""
    echo "------------------------------------------"
    echo ">>> 第 $i 次运行 (共 $RUN_COUNT 次)..."
    echo "------------------------------------------"
    
    # 运行 SLAM (屏蔽详细输出，只显示错误)
    start_time=$(date +%s)
    ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml "$DATASET_PATH" Examples/RGB-D/fr3_sitting_static.txt > /dev/null 2>&1
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "   ⏱️  运行耗时: $duration 秒"
    
    # 检查是否生成了轨迹文件
    if [ ! -f "$RESULT_DIR/KeyFrameTrajectory.txt" ] || [ ! -f "$RESULT_DIR/CameraTrajectory.txt" ]; then
        echo "❌ 错误: 第 $i 次运行失败，未生成轨迹文件!"
        continue
    fi
    
    # 重命名轨迹文件
    cp "$RESULT_DIR/CameraTrajectory.txt" "$RESULT_DIR/CameraTrajectory_$i.txt"
    cp "$RESULT_DIR/KeyFrameTrajectory.txt" "$RESULT_DIR/KeyFrameTrajectory_$i.txt"
    
    # --- 1. 计算 ATE (KeyFrameTrajectory) ---
    echo "📊 计算 ATE RMSE (KeyFrames)..."
    evo_ape tum "$GROUNDTRUTH" "$RESULT_DIR/KeyFrameTrajectory_$i.txt" -va > "$RESULT_DIR/ate_out_$i.txt" 2>&1
    ate_rmse=$(grep "rmse" "$RESULT_DIR/ate_out_$i.txt" | awk '{print $NF}')
    
    if [ -z "$ate_rmse" ]; then
         echo "❌ 无法提取 ATE!"
         cat "$RESULT_DIR/ate_out_$i.txt"
         ate_rmse=0
    else
         echo "✅ ATE (RMSE): $ate_rmse m"
    fi
    
    # --- 2. 计算 RPE (CameraTrajectory) ---
    echo "📊 计算 RPE RMSE (Camera, delta=0.5m)..."
    evo_rpe tum "$GROUNDTRUTH" "$RESULT_DIR/CameraTrajectory_$i.txt" -va --delta 0.5 --delta_unit m > "$RESULT_DIR/rpe_out_$i.txt" 2>&1
    # 检查 rpe_out 内容确保运行完成
    if [ ! -s "$RESULT_DIR/rpe_out_$i.txt" ]; then
        echo "❌ RPE 计算无输出或被中断"
        rpe_rmse=0
    else
        rpe_rmse=$(grep "rmse" "$RESULT_DIR/rpe_out_$i.txt" | awk '{print $NF}')
        
        if [ -z "$rpe_rmse" ]; then
             echo "❌ 无法提取 RPE!"
             cat "$RESULT_DIR/rpe_out_$i.txt"
             rpe_rmse=0
        else
             echo "✅ RPE (RMSE, 0.5m Drift): $rpe_rmse m"
        fi
    fi

    # --- 3. 计算 RPE Rotation (CameraTrajectory) ---
    echo "📊 计算 RPE Rotation RMSE (Camera, delta=0.5m)..."
    evo_rpe tum "$GROUNDTRUTH" "$RESULT_DIR/CameraTrajectory_$i.txt" -va --pose_relation angle_deg --delta 0.5 --delta_unit m > "$RESULT_DIR/rpe_rot_out_$i.txt" 2>&1
    # 检查 rpe_rot_out 内容确保运行完成
    if [ ! -s "$RESULT_DIR/rpe_rot_out_$i.txt" ]; then
        echo "❌ RPE Rotation 计算无输出或被中断"
        rpe_rot_rmse=0
    else
        rpe_rot_rmse=$(grep "rmse" "$RESULT_DIR/rpe_rot_out_$i.txt" | awk '{print $NF}')
        
        if [ -z "$rpe_rot_rmse" ]; then
             echo "❌ 无法提取 RPE Rotation!"
             cat "$RESULT_DIR/rpe_rot_out_$i.txt"
             rpe_rot_rmse=0
        else
             echo "✅ RPE Rotation (RMSE, 0.5m Drift): $rpe_rot_rmse deg"
        fi
    fi

    # 累加结果
    if [ $(echo "$ate_rmse > 0" | bc -l) -eq 1 ] && [ $(echo "$rpe_rmse > 0" | bc -l) -eq 1 ] && [ $(echo "$rpe_rot_rmse > 0" | bc -l) -eq 1 ]; then
        total_ate_rmse=$(python3 -c "print($total_ate_rmse + $ate_rmse)")
        total_rpe_rmse=$(python3 -c "print($total_rpe_rmse + $rpe_rmse)")
        total_rpe_rot_rmse=$(python3 -c "print($total_rpe_rot_rmse + $rpe_rot_rmse)")
        count_success=$((count_success + 1))
    fi

done

# 计算平均值
if [ "$count_success" -gt 0 ]; then
    avg_ate=$(python3 -c "print($total_ate_rmse / $count_success)")
    avg_rpe=$(python3 -c "print($total_rpe_rmse / $count_success)")
    avg_rpe_rot=$(python3 -c "print($total_rpe_rot_rmse / $count_success)")
    
    echo ""
    echo "=========================================="
    echo "📈 测试结果汇总 ($count_success / $RUN_COUNT 次成功)"
    echo "=========================================="
    echo "平均 ATE RMSE (KeyFrames): $avg_ate m"
    echo "平均 RPE RMSE (Camera Translation): $avg_rpe m"
    echo "平均 RPE RMSE (Camera Rotation):    $avg_rpe_rot deg"
    echo "=========================================="
    echo "详细图表: $RESULT_DIR/ate_kf_*.png, rpe_cam_*.png"
else
    echo "未成功运行任何测试。"
fi
