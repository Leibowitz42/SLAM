#pragma once

#include <cstdint>
#include <cstddef>

/**
 * GPU Preprocessing Pipeline for YOLO inference on Jetson / x86 CUDA.
 *
 * Performs in a single kernel launch:
 *   1. Bilinear resize
 *   2. LetterBox padding (gray 114/255)
 *   3. BGR -> RGB color conversion
 *   4. Normalize to [0, 1] (divide by 255)
 *   5. HWC -> CHW layout transformation
 *
 * NOTE: This header is intentionally free of any CUDA headers so it can be
 *       safely included by regular .cpp files compiled with g++.
 */

struct GpuPreprocessContext {
    // Device memory pointers (allocated once, reused every frame)
    // These are actually cudaStream_t / device pointers, but declared as void*
    // to avoid requiring cuda_runtime.h in user code.
    void* d_src = nullptr;       // Source image on GPU (BGR, HWC, uint8)
    float* d_dst = nullptr;      // Output tensor on GPU (RGB, CHW, normalized)

    // Source image dimensions (updated per frame)
    int src_width = 0;
    int src_height = 0;

    // Network input dimensions (fixed)
    int dst_width = 640;
    int dst_height = 640;

    // Maximum source buffer size (to avoid reallocation)
    size_t max_src_bytes = 0;

    // CUDA stream (opaque handle, actually cudaStream_t)
    void* stream = nullptr;

    // Letterbox params (output, computed by preprocess)
    float ratio = 1.0f;       // scale ratio
    float pad_w = 0.0f;       // left padding in pixels
    float pad_h = 0.0f;       // top padding in pixels
};

/**
 * Initialize the GPU preprocess context.
 * Allocates device memory for the maximum expected image size.
 */
void gpuPreprocessInit(GpuPreprocessContext& ctx,
                       int max_width, int max_height,
                       int net_width = 640, int net_height = 640);

/**
 * Run the GPU preprocessing pipeline.
 * Uploads the source image to GPU, runs the fused kernel, output stays on GPU.
 *
 * After this call:
 *   - ctx.d_dst points to the CHW float tensor on GPU
 *   - ctx.ratio, ctx.pad_w, ctx.pad_h contain LetterBox params for postprocessing
 */
void gpuPreprocessExecute(GpuPreprocessContext& ctx,
                          const uint8_t* src_data,
                          int src_width, int src_height, int src_step);

/**
 * Destroy the GPU preprocess context and free device memory.
 */
/**
 * Execute GPU preprocessing directly routing output into an external Device Pointer.
 * (Bypasses intermediate D2D buffers, while leveraging a hardware-aware H2D DMA selection
 * to prevent shared memory bus contention on Jetson platforms)
 */
void gpuPreprocessExecuteDirect(GpuPreprocessContext& ctx,
                                const unsigned char* src_data,
                                int src_width, int src_height, int src_step,
                                float* external_dst_ptr);

void gpuPreprocessDestroy(GpuPreprocessContext& ctx);
