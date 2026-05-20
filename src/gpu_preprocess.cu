#include <cuda_runtime.h>
#include "gpu_preprocess.h"
#include <iostream>
#include <algorithm>
#include <cstring>

/**
 * Fused CUDA kernel: Resize(bilinear) + LetterBox + BGR->RGB + Normalize + HWC->CHW
 *
 * Each thread computes ONE pixel in the output (dst_width x dst_height).
 * It reverse-maps to the source image, performs bilinear interpolation,
 * and writes the result directly in CHW float format.
 */
__global__ void preprocessKernel(
    const uint8_t* __restrict__ src,   // Source image: BGR, HWC, uint8
    float*         __restrict__ dst,   // Output tensor: RGB, CHW, float32
    int src_width, int src_height, int src_step,
    int dst_width, int dst_height,
    float ratio,                       // scale ratio (same for w and h)
    float pad_w,                       // left padding
    float pad_h                        // top padding
)
{
    // Each thread handles one (dst_x, dst_y) pixel
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_width || dst_y >= dst_height)
        return;

    const int dst_area = dst_width * dst_height;

    // Reverse map: from dst pixel to src pixel
    float src_xf = ((float)dst_x - pad_w) / ratio;
    float src_yf = ((float)dst_y - pad_h) / ratio;

    // Check if this pixel is in the padding region
    if (src_xf < -0.5f || src_xf >= (float)src_width - 0.5f ||
        src_yf < -0.5f || src_yf >= (float)src_height - 0.5f)
    {
        // Padding region: fill with 114/255 ≈ 0.447
        const float pad_val = 114.0f / 255.0f;
        dst[0 * dst_area + dst_y * dst_width + dst_x] = pad_val;  // R
        dst[1 * dst_area + dst_y * dst_width + dst_x] = pad_val;  // G
        dst[2 * dst_area + dst_y * dst_width + dst_x] = pad_val;  // B
        return;
    }

    // Bilinear interpolation coordinates
    int x0 = (int)floorf(src_xf);
    int y0 = (int)floorf(src_yf);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Clamp to image boundaries
    x0 = max(0, min(x0, src_width - 1));
    y0 = max(0, min(y0, src_height - 1));
    x1 = max(0, min(x1, src_width - 1));
    y1 = max(0, min(y1, src_height - 1));

    float dx = src_xf - floorf(src_xf);
    float dy = src_yf - floorf(src_yf);

    // Weights for bilinear interpolation
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = dx * (1.0f - dy);
    float w10 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    // Read 4 source pixels (BGR format)
    const uint8_t* p00 = src + y0 * src_step + x0 * 3;
    const uint8_t* p01 = src + y0 * src_step + x1 * 3;
    const uint8_t* p10 = src + y1 * src_step + x0 * 3;
    const uint8_t* p11 = src + y1 * src_step + x1 * 3;

    // Interpolate each channel, convert BGR->RGB, normalize by 255
    // Source is BGR: [0]=B, [1]=G, [2]=R
    // Output is RGB CHW: plane0=R, plane1=G, plane2=B
    float b = (w00 * p00[0] + w01 * p01[0] + w10 * p10[0] + w11 * p11[0]) / 255.0f;
    float g = (w00 * p00[1] + w01 * p01[1] + w10 * p10[1] + w11 * p11[1]) / 255.0f;
    float r = (w00 * p00[2] + w01 * p01[2] + w10 * p10[2] + w11 * p11[2]) / 255.0f;

    // Write in CHW layout: [C, H, W]
    int pixel_idx = dst_y * dst_width + dst_x;
    dst[0 * dst_area + pixel_idx] = r;  // R channel
    dst[1 * dst_area + pixel_idx] = g;  // G channel
    dst[2 * dst_area + pixel_idx] = b;  // B channel
}


void gpuPreprocessInit(GpuPreprocessContext& ctx,
                       int max_width, int max_height,
                       int net_width, int net_height)
{
    ctx.dst_width = net_width;
    ctx.dst_height = net_height;
    ctx.max_src_bytes = (size_t)max_width * max_height * 3;

    // Allocate source buffer on GPU (for uploading each frame)
    uint8_t* d_src_ptr = nullptr;
    cudaMalloc(&d_src_ptr, ctx.max_src_bytes);
    ctx.d_src = d_src_ptr;

    // Allocate output tensor buffer on GPU: 1 x 3 x H x W float32
    size_t dst_bytes = (size_t)3 * net_width * net_height * sizeof(float);
    float* d_dst_ptr = nullptr;
    cudaMalloc(&d_dst_ptr, dst_bytes);
    ctx.d_dst = d_dst_ptr;

    // Create a CUDA stream for async H2D copy + kernel launch
    cudaStream_t cuda_stream;
    cudaStreamCreate(&cuda_stream);
    ctx.stream = cuda_stream;

    std::cout << "[GPU Preprocess] Initialized: max_src=" << max_width << "x" << max_height
              << ", net=" << net_width << "x" << net_height
              << ", src_buf=" << ctx.max_src_bytes / 1024 << "KB"
              << ", dst_buf=" << dst_bytes / 1024 << "KB" << std::endl;
}


void gpuPreprocessExecuteDirect(GpuPreprocessContext& ctx,
                                const uint8_t* src_data,
                                int src_width, int src_height, int src_step,
                                float* external_dst_ptr)
{
    ctx.src_width = src_width;
    ctx.src_height = src_height;

    cudaStream_t cuda_stream = (cudaStream_t)ctx.stream;

    // Compute LetterBox parameters (same logic as CPU LetterBox)
    float r_w = (float)ctx.dst_width / (float)src_width;
    float r_h = (float)ctx.dst_height / (float)src_height;
    ctx.ratio = std::min(r_w, r_h);

    int new_unpad_w = (int)(src_width * ctx.ratio + 0.5f);
    int new_unpad_h = (int)(src_height * ctx.ratio + 0.5f);

    ctx.pad_w = (ctx.dst_width - new_unpad_w) / 2.0f;
    ctx.pad_h = (ctx.dst_height - new_unpad_h) / 2.0f;

    // Upload source image to GPU (async) only if it's not already a CUDA device/managed pointer
    size_t src_bytes = (size_t)src_step * src_height;
    
    bool is_device_accessible = false;
    cudaPointerAttributes attributes;
    // In CUDA, if pointer is not registered, this returns an error which we can ignore.
    if (cudaPointerGetAttributes(&attributes, src_data) == cudaSuccess) {
        // We only bypass the H2D copy if the memory is physically on the device or managed.
        // If it is host memory (cudaMemoryTypeHost) like pinned memory allocated by CudaPinnedAllocator,
        // it still resides on the host, so we must upload it to device memory to avoid illegal memory access or severe bus contention.
        if (attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged) {
            is_device_accessible = true;
        }
    } else {
        cudaGetLastError(); // Clear the error from the driver
    }

    const uint8_t* kernel_src = src_data;

    if (!is_device_accessible) {
        if (src_bytes > ctx.max_src_bytes) {
            std::cerr << "[GPU Preprocess] WARNING: Reallocating src buffer from "
                      << ctx.max_src_bytes << " to " << src_bytes << " bytes" << std::endl;
            cudaFree(ctx.d_src);
            ctx.max_src_bytes = src_bytes;
            uint8_t* d_src_ptr = nullptr;
            cudaMalloc(&d_src_ptr, ctx.max_src_bytes);
            ctx.d_src = d_src_ptr;
        }

        cudaMemcpyAsync(ctx.d_src, src_data, src_bytes,
                        cudaMemcpyHostToDevice, cuda_stream);
        kernel_src = (const uint8_t*)ctx.d_src;
    }

    dim3 block(32, 32);
    dim3 grid((ctx.dst_width + block.x - 1) / block.x,
              (ctx.dst_height + block.y - 1) / block.y);

    // Pass the correct pointer to swallow the compute result directly
    preprocessKernel<<<grid, block, 0, cuda_stream>>>(
        kernel_src, external_dst_ptr,
        src_width, src_height, src_step,
        ctx.dst_width, ctx.dst_height,
        ctx.ratio, ctx.pad_w, ctx.pad_h
    );

    // NOT performing cudaStreamSynchronize here explicitly to let TensorRT stack operations!
}

void gpuPreprocessExecute(GpuPreprocessContext& ctx,
                          const uint8_t* src_data,
                          int src_width, int src_height, int src_step)
{
    ctx.src_width = src_width;
    ctx.src_height = src_height;

    cudaStream_t cuda_stream = (cudaStream_t)ctx.stream;

    // Compute LetterBox parameters (same logic as CPU LetterBox)
    float r_w = (float)ctx.dst_width / (float)src_width;
    float r_h = (float)ctx.dst_height / (float)src_height;
    ctx.ratio = std::min(r_w, r_h);

    int new_unpad_w = (int)(src_width * ctx.ratio + 0.5f);
    int new_unpad_h = (int)(src_height * ctx.ratio + 0.5f);

    ctx.pad_w = (ctx.dst_width - new_unpad_w) / 2.0f;
    ctx.pad_h = (ctx.dst_height - new_unpad_h) / 2.0f;

    // Upload source image to GPU (async)
    size_t src_bytes = (size_t)src_step * src_height;
    if (src_bytes > ctx.max_src_bytes) {
        // Reallocate if image is larger than expected
        std::cerr << "[GPU Preprocess] WARNING: Reallocating src buffer from "
                  << ctx.max_src_bytes << " to " << src_bytes << " bytes" << std::endl;
        cudaFree(ctx.d_src);
        ctx.max_src_bytes = src_bytes;
        uint8_t* d_src_ptr = nullptr;
        cudaMalloc(&d_src_ptr, ctx.max_src_bytes);
        ctx.d_src = d_src_ptr;
    }

    cudaMemcpyAsync(ctx.d_src, src_data, src_bytes,
                    cudaMemcpyHostToDevice, cuda_stream);

    // Launch the fused preprocessing kernel
    dim3 block(32, 32);
    dim3 grid((ctx.dst_width + block.x - 1) / block.x,
              (ctx.dst_height + block.y - 1) / block.y);

    preprocessKernel<<<grid, block, 0, cuda_stream>>>(
        (const uint8_t*)ctx.d_src, ctx.d_dst,
        src_width, src_height, src_step,
        ctx.dst_width, ctx.dst_height,
        ctx.ratio, ctx.pad_w, ctx.pad_h
    );

    // Synchronize to ensure the output is ready
    cudaStreamSynchronize(cuda_stream);
}


void gpuPreprocessDestroy(GpuPreprocessContext& ctx)
{
    if (ctx.d_src) {
        cudaFree(ctx.d_src);
        ctx.d_src = nullptr;
    }
    if (ctx.d_dst) {
        cudaFree(ctx.d_dst);
        ctx.d_dst = nullptr;
    }
    if (ctx.stream) {
        cudaStreamDestroy((cudaStream_t)ctx.stream);
        ctx.stream = nullptr;
    }
    std::cout << "[GPU Preprocess] Destroyed." << std::endl;
}
