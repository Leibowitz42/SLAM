---
name: Jetson SLAM System Optimization
description: Techniques and pipelines for optimizing visual SLAM systems on Jetson platforms using GPU preprocessing via CUDA and unified CMake C++ polymorphism.
---

# Jetson SLAM Optimization Skill

## 1. Goal
When dealing with visual SLAM systems (like ORB-SLAM3) deployed on NVIDIA Jetson or edge GPU systems alongside heavy deep-learning visual segmenters (like YOLOv8-seg), raw CPU processing can severely bottleneck the camera stream frame rate. This skill defines how to deploy our "Zero-Touch CPU" optimization architecture.

## 2. GPU Preprocessing (Zero-Copy VRAM Pipeline)
Do not use `cv::cuda` or `cudaHostAlloc` (Mapped memory) for preprocessing frames on memory-constrained devices (like Jetson Orin Nano 4GB). It leads to severe bus contention and CPU L2 Cache eviction.

Instead, implement native Fused CUDA Kernels that compress:
- Bilinear Interpolation (Resize)
- Padding / LetterBox
- Color conversion (BGR -> RGB)
- Normalization (/255.f)
- Layout transformation (HWC -> CHW)

### Usage Pattern
1. Allocate source `d_src` (BGR image) and destination `d_dst` (Float CHW Tensor) ONCE in VRAM.
2. `cudaMemcpyAsync` the raw camera `uint8_t` feed to `d_src`.
3. Launch the `preprocessKernel` to construct the `CHW` array directly on `d_dst`.
4. Pass `d_dst` directly to ONNX Runtime via `Ort::MemoryInfo("Cuda")` or to TensorRT's context bindings via DeviceToDevice memory transfers.

## 3. Opaque Pointers Context Management
Header files (e.g. `gpu_preprocess.h`) should NEVER include `<cuda_runtime.h>` to avoid compilation collision with generic C++ compilers handling other SLAM headers like `sophus` or `g2o`. 
Wrap CUDA streams and pointers as `void*` in the context header, and `reinterpret_cast` them inwardly inside `.cu` compiled by `nvcc`.

## 4. CMake Polymorphism for Multi-Platform
To maintain a unified repository across an x86 PC (ONNX CPU/GPU) and Edge Devices (Native TensorRT):
- Abstract the model implementation into a base interface (e.g. `IYolov8Seg`).
- In `CMakeLists.txt`, use dynamic `find_path()` and `find_library()` for `NvInferRuntime.h` and `cudart`. 
- Define a macro `-DUSE_TENSORRT` ONLY if the environment possesses Jetson's TRT hardware headers.
- Dynamically detect `.onnx` or `.engine` extension patterns to polymorphically boot the exact acceleration class.

## 5. Zero-Touch CPU Postprocessing
Only process the resulting Bounding Boxes and Masks (NMS) on the CPU after bringing back lightweight tensors. Avoid mapping large tensor gradients unless explicitly needed. Calculate offset coordinates dynamically from Letterbox scaling indices preserved locally in memory by the C++ `GpuPreprocessContext`.
