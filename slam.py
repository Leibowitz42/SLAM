import matplotlib.pyplot as plt
import re
import sys
import os
import numpy as np

# 设置非交互式后端
import matplotlib
matplotlib.use('Agg')

def parse_tegrastats(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return None

    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            ram = re.search(r'RAM (\d+)', line)
            gpu = re.search(r'GR3D_FREQ (\d+)%', line)
            cpu_part = re.search(r'CPU \[(.*?)\]', line)
            
            if ram and gpu and cpu_part:
                cpu_data = re.findall(r'(\d+)%', cpu_part.group(1))
                avg_cpu = sum(map(int, cpu_data)) / len(cpu_data) if cpu_data else 0
                data.append({
                    'index': i,
                    'ram': int(ram.group(1)),
                    'gpu': int(gpu.group(1)),
                    'cpu': avg_cpu
                })
    return data

def analyze_and_plot(data, input_file):
    if not data:
        print("未发现有效采样。")
        return

    ram = np.array([d['ram'] for d in data])
    gpu = np.array([d['gpu'] for d in data])
    cpu = np.array([d['cpu'] for d in data])
    t = np.arange(len(data)) * 0.5 # 500ms 一个采样

    # 简单的运行区间检测：当 GPU 有负载或 CPU 负载显著持续升高时
    # 结合 tegrastats 记录，SLAM 运行时 GPU 负载会从 0% 跳跃
    is_running = (gpu > 2) | (cpu > 40)
    # 找到运行区间的起点和终点
    indices = np.where(is_running)[0]
    start_idx, end_idx = (indices[0], indices[-1]) if len(indices) > 0 else (0, len(t)-1)

    print(f"检测到 SLAM 运行区间: 第 {start_idx} 采样 - 第 {end_idx} 采样 (约 {t[start_idx]:.1f}s - {t[end_idx]:.1f}s)")
    print(f"运行期间平均 GPU 负载: {np.mean(gpu[start_idx:end_idx+1]):.1f}%")
    print(f"运行期间最高 RAM 占用: {np.max(ram[start_idx:end_idx+1])} MB")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    metrics = [
        (ram, 'RAM (MB)', 'r', 'System Memory Usage'),
        (gpu, 'GPU (%)', 'g', 'NVIDIA GPU Load'),
        (cpu, 'Avg CPU (%)', 'b', 'Multi-core CPU Load')
    ]

    for i, (val, ylabel, color, title) in enumerate(metrics):
        axes[i].plot(t, val, color=color, linewidth=1.5, label='Actual Data')
        # 高亮运行区间
        axes[i].axvspan(t[start_idx], t[end_idx], color='cyan', alpha=0.15, label='SLAM Activity')
        axes[i].set_ylabel(ylabel)
        axes[i].grid(True, linestyle='--', alpha=0.5)
        if i == 0: axes[i].set_title(f"SLAM Performance Audit: {input_file}")

    axes[2].set_xlabel('Time (seconds)')
    axes[2].legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('performance_report.png')
    print("区分完成！分析报告 `performance_report.png` 已更新。")

if __name__ == "__main__":
    file = sys.argv[1] if len(sys.argv) > 1 else 'slam_load.txt'
    data = parse_tegrastats(file)
    analyze_and_plot(data, file)