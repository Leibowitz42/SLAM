#pragma once

#include <opencv2/core.hpp>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

class CudaPinnedAllocator : public cv::MatAllocator {
public:
    cv::UMatData* allocate(int dims, const int* sizes, int type,
                           void* data, size_t* step,
                           cv::AccessFlag flags, cv::UMatUsageFlags usageFlags) const override {
        size_t total = CV_ELEM_SIZE(type);
        for (int i = dims - 1; i >= 0; i--) {
            if (step) {
                if (data && step[i] != cv::Mat::AUTO_STEP) {
                    CV_Assert(total <= step[i]);
                    total = step[i];
                }
                else {
                    step[i] = total;
                }
            }
            total *= sizes[i];
        }

        cv::UMatData* u = new cv::UMatData(this);
        u->size = total;

        if (data) {
            u->data = u->origdata = static_cast<uchar*>(data);
            u->flags |= cv::UMatData::USER_ALLOCATED;
        } else {
            // =====================================================================
            // [NOVELTY 2]: Zero-Copy Memory Allocation for UMA Architecture
            // =====================================================================
            // On Jetson (Unified Memory Architecture), CPU and GPU share physical RAM.
            // WARNING: Using `cudaHostAllocMapped` makes the memory UNCACHED by the CPU,
            // which causes OpenCV's ORB extraction (CPU-based) to be brutally slow (100ms+).
            // SOLUTION: We use `cudaMallocManaged` (Unified Memory). On Jetson Xavier/Orin, 
            // Unified Memory is CPU-CACHED and uses hardware cache coherency. 
            // This gives us full CPU speed for ORB, and 0-copy DMA for YOLO TensorRT!
            // 
            // [OOM FIX]: ORB-SLAM3 creates millions of tiny cv::Mat (for math, descriptors).
            // Allocating them via CUDA exhausts Jetson's mmap limits. We ONLY use Unified Memory 
            // for large matrices (like 640x480 images > 100KB).
            if (total >= 100000) {
                void* ptr = nullptr;
                cudaError_t err = cudaMallocManaged(&ptr, total, cudaMemAttachGlobal);
                if (err != cudaSuccess) {
                    std::cerr << "cudaMallocManaged failed: " << cudaGetErrorString(err) << " (Size: " << total << ")" << std::endl;
                    ptr = malloc(total);
                    u->userdata = (void*)0; // Mark as standard malloc
                } else {
                    u->userdata = (void*)1; // Mark as cudaMallocManaged
                }
                u->data = u->origdata = static_cast<uchar*>(ptr);
            } else {
                // Fast path for small matrices (math, descriptors, UI)
                u->data = u->origdata = static_cast<uchar*>(malloc(total));
                u->userdata = (void*)0; // Mark as standard malloc
            }
        }

        return u;
    }

    bool allocate(cv::UMatData* u, cv::AccessFlag accessFlags, cv::UMatUsageFlags usageFlags) const override {
        return false;
    }

    void deallocate(cv::UMatData* u) const override {
        if (!u) return;
        
        if ((u->flags & cv::UMatData::USER_ALLOCATED) == 0) {
            if (u->origdata) {
                if (u->userdata == (void*)1) {
                    cudaFree(u->origdata);
                } else {
                    free(u->origdata);
                }
            }
        }
        delete u;
    }
};

static CudaPinnedAllocator g_cudaAllocator;

inline void SetCudaPinnedAllocator() {
    cv::Mat::setDefaultAllocator(&g_cudaAllocator);
    std::cout << "[CUDA] OpenCV Default Allocator set to CudaPinnedAllocator!" << std::endl;
}

