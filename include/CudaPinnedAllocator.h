#pragma once

#include <opencv2/core.hpp>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <unordered_set>
#include <mutex>

class CudaPinnedAllocator : public cv::MatAllocator {
private:
    mutable std::unordered_set<void*> managed_ptrs;
    mutable std::mutex mtx;

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
            // [OOM FIX]: Only allocate large matrices via CUDA Managed Memory
            if (total >= 100000) {
                void* ptr = nullptr;
                // Use cudaHostAllocDefault to ensure CPU caching on Jetson while keeping it Pinned.
                cudaError_t err = cudaHostAlloc(&ptr, total, cudaHostAllocDefault);
                if (err != cudaSuccess) {
                    std::cerr << "cudaHostAlloc failed: " << cudaGetErrorString(err) << " (Size: " << total << ")" << std::endl;
                    ptr = malloc(total);
                } else {
                    std::lock_guard<std::mutex> lock(mtx);
                    managed_ptrs.insert(ptr);
                }
                u->data = u->origdata = static_cast<uchar*>(ptr);
            } else {
                u->data = u->origdata = static_cast<uchar*>(malloc(total));
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
                bool is_managed = false;
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    if (managed_ptrs.find(u->origdata) != managed_ptrs.end()) {
                        is_managed = true;
                        managed_ptrs.erase(u->origdata);
                    }
                }

                if (is_managed) {
                    cudaFreeHost(u->origdata);
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

