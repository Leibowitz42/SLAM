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
            // Instead of standard malloc/new, we allocate OpenCV Mat data directly 
            // on the CUDA Host using cudaHostAlloc with the cudaHostAllocMapped flag.
            // On Jetson (Unified Memory Architecture), CPU and GPU share physical RAM.
            // Pinned (Page-locked) memory prevents the OS from paging out the buffer,
            // and the 'Mapped' flag exposes this exact physical address space to 
            // the GPU's page tables. This entirely eliminates expensive Host-to-Device 
            // (H2D) and Device-to-Host (D2H) cudaMemcpy operations during YOLO inference.
            void* ptr = nullptr;
            cudaError_t err = cudaHostAlloc(&ptr, total, cudaHostAllocMapped);
            if (err != cudaSuccess) {
                std::cerr << "cudaHostAlloc failed: " << cudaGetErrorString(err) << std::endl;
                // Fallback to standard allocation
                ptr = malloc(total);
            }
            u->data = u->origdata = static_cast<uchar*>(ptr);
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
                cudaError_t err = cudaFreeHost(u->origdata);
                if (err != cudaSuccess) {
                    // It might have been allocated with malloc as fallback
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

