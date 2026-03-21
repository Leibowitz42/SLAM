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
            // Use cudaHostAlloc for pinned memory allocation
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

