/* Last Updated: 24.08.27. 18:30 */
#include "layer.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

/* Linear
 * GPU 병렬화: 각 출력 요소를 병렬로 계산합니다.
 * half 정밀도 활용: GPU의 native half 타입을 사용하여 연산 속도를 향상시킵니다.
 */
__global__ void LinearKernel(half *in, half *w, half *b, half *out,
                             size_t M, size_t N, size_t K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N) {
        half sum = __float2half(0.0f);
        for (size_t k = 0; k < K; k++) {
            sum = __hadd(sum, __hmul(in[m * K + k], w[n * K + k]));
        }
        out[m * N + n] = __hadd(sum, b[n]);
    }
}

void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out, cudaStream_t stream) {
    size_t M = out->shape[0];
    size_t N = out->shape[1];
    size_t K = w->shape[1];

    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    LinearKernel<<<gridDim, blockDim, 0, stream>>>(in->d_buf, w->d_buf, b->d_buf, out->d_buf, M, N, K);
}

/* Reshape
 * @param [in]   in: [N, D]
 * @param [out] out: [N, C, H, W]
 */
__global__ void ReshapeKernel(half *in, half *out,
                              size_t N, size_t D, size_t C, size_t H, size_t W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C * H * W) {
        size_t n = idx / (C * H * W);
        size_t chw = idx % (C * H * W);
        out[idx] = in[n * D + chw];
    }
}

void Reshape(Tensor *in, Tensor *out, cudaStream_t stream) {
    size_t N = in->shape[0];
    size_t D = in->shape[1];
    size_t C = out->shape[1];
    size_t H = out->shape[2];
    size_t W = out->shape[3];

    int totalThreads = N * C * H * W;
    int blockSize = 256;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;
    ReshapeKernel<<<numBlocks, blockSize, 0, stream>>>(in->d_buf, out->d_buf, N, D, C, H, W);
}

/* ConvTranspose2d
 * @param [in1]     in: [N, C, H, W]
 * @param [in2] weight: [C, K, R, S]
 * @param [in3]   bias: [K]
 * @param [out]    out: [N, K, OH, OW]
 */


__global__ void ConvTranspose2dKernel(const half* __restrict__ in,
                                      const half* __restrict__ weight,
                                      const half* __restrict__ bias,
                                      half* __restrict__ out,
                                      int N, int C, int H, int W,
                                      int K, int R, int S, int OH, int OW,
                                      int stride, int pad, int dilation) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;
    if (k >= K || oh >= OH || ow >= OW) return;
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            int h = (oh + pad - r * dilation) / stride;
            int w = (ow + pad) / stride;
            if (h >= 0 && h < H &&
                (oh + pad - r * dilation) % stride == 0) {
#pragma unroll 4
                for (int s = 0; s < S; ++s) {
                    if (w >= 0 && w < W &&
                        (ow + pad - s * dilation) % stride == 0) {
                        float in_val = __half2float(in[c * H * W + h * W + w]);
                        float weight_val = __half2float(weight[c * K * R * S + k * R * S + r * S + s]);
                        sum += in_val * weight_val;
                    }
                    w = (ow + pad - (s + 1) * dilation) / stride;
                }
            }
        }
    }
    sum += __half2float(bias[k]);
    out[k * OH * OW + oh * OW + ow] = __float2half(sum);
}


void ConvTranspose2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out, cudaStream_t stream) {
    size_t N = in->shape[0];
    size_t C = in->shape[1];
    size_t H = in->shape[2];
    size_t W = in->shape[3];
    size_t K = weight->shape[1];
    size_t R = weight->shape[2];
    size_t S = weight->shape[3];
    size_t OH = out->shape[2];
    size_t OW = out->shape[3];

    const size_t stride = 2;
    const size_t pad = 1;
    const size_t dilation = 1;

    dim3 blockDim(16, 16, 16);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x,
                 (OH + blockDim.y - 1) / blockDim.y,
                 (OW + blockDim.z - 1) / blockDim.z);

    ConvTranspose2dKernel<<<gridDim, blockDim, 0, stream>>>(
            in->d_buf, weight->d_buf, bias->d_buf, out->d_buf,
            N, C, H, W, K, R, S, OH, OW,
            stride, pad, dilation);
}

/* BatchNorm2d (track_running_stats=False)
 * @param [in1]     in: [N, C, H, W]
 * @param [in2] weight: [C]
 * @param [in3]   bias: [C]
 * @param [out]    out: [N, C, H, W]
 */
__global__ void BatchNorm2d_kernel(half *in, half *weight, half *bias, half *out,
                                   size_t N, size_t C, size_t H, size_t W) {
    const float eps = 1e-5f;
    size_t c = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;
    size_t HW = H * W;

    __shared__ float mean;
    __shared__ float var;

    // Step 1: Calculate mean
    float sum = 0.0f;
    for (size_t i = tid; i < HW; i += stride) {
        sum += __half2float(in[c * HW + i]);
    }

    __shared__ float block_sum;
    block_sum = 0.0f;
    __syncthreads();

    atomicAdd(&block_sum, sum);
    __syncthreads();

    if (tid == 0) {
        mean = block_sum / (float)HW;
    }
    __syncthreads();

    // Step 2: Calculate variance
    float var_sum = 0.0f;
    for (size_t i = tid; i < HW; i += stride) {
        float diff = __half2float(in[c * HW + i]) - mean;
        var_sum += diff * diff;
    }

    block_sum = 0.0f;
    __syncthreads();

    atomicAdd(&block_sum, var_sum);
    __syncthreads();

    if (tid == 0) {
        var = block_sum / (float)HW;
    }
    __syncthreads();

    // Step 3: Normalize and scale
    float w = __half2float(weight[c]);
    float b = __half2float(bias[c]);
    float invstd = rsqrtf(var + eps);

    for (size_t i = tid; i < HW; i += stride) {
        float normalized = (__half2float(in[c * HW + i]) - mean) * invstd;
        out[c * HW + i] = __float2half(w * normalized + b);
    }
}

void BatchNorm2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out, cudaStream_t stream) {
    size_t N = in->shape[0];
    size_t C = in->shape[1];
    size_t H = in->shape[2];
    size_t W = in->shape[3];

    dim3 grid(C);  // One block per channel
    dim3 block(256);  // Adjust this based on your GPU capabilities
    BatchNorm2d_kernel<<<grid, block, 0, stream>>>(in->d_buf, weight->d_buf, bias->d_buf, out->d_buf, N, C, H, W);
}

/* LeakyReLU GPU kernel
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
__global__ void LeakyReLU_kernel(half *inout, size_t N, half alpha) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (inout[idx] < half(0)) { inout[idx] *= alpha; }
    }
}

/* LeakyReLU using CUDA GPU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void LeakyReLU(Tensor *inout, cudaStream_t stream) {
    size_t N = inout->num_elem();
    const half alpha = 0.01;

    LeakyReLU_kernel<<<(N + 255) / 256, 256, 0, stream>>>(inout->d_buf, N, alpha);
}

/* Conv2d
 * @param [in1]     in: [N, C, H, W]
 * @param [in2] weight: [K, C, R, S]
 * @param [in3]   bias: [K]
 * @param [out]    out: [N, K, OH, OW]
 */

__global__ void Conv2d_kernel(half *in, half *weight, half *bias, half *out,
                              size_t N, size_t C, size_t H, size_t W,
                              size_t K, size_t R, size_t S,
                              size_t OH, size_t OW,
                              size_t stride, size_t pad, size_t dilation) {
    int n = blockIdx.x;
    int k = blockIdx.y;
    int oh_block = blockIdx.z / ((OW + blockDim.y - 1) / blockDim.y);
    int ow_block = blockIdx.z % ((OW + blockDim.y - 1) / blockDim.y);
    int oh = oh_block * blockDim.x + threadIdx.x;
    int ow = ow_block * blockDim.y + threadIdx.y;
    if (oh < OH && ow < OW) {
        half sum = bias[k];
        for (int c = 0; c < C; c++) {
            for (int r = 0; r < R; r++) {
                int h = oh * stride - pad + r * dilation;
                int w = ow * stride - pad;
                if (h >= 0 && h < H) {
                    //if (S >= 4) {
                    //if (-1) {
#pragma unroll
                    for (int s = 0; s < 4; s++) {
                        if (w >= 0 && w < W) {
                            sum = __hadd(sum, __hmul(in[n * C * H * W + c * H * W + h * W + w],
                                                     weight[k * C * R * S + c * R * S + r * S + s]));
                        }
                        w += dilation;
                    }
                    // } else {
                    //     for (int s = 0; s < S; s++) {
                    //         if (w >= 0 && w < W) {
                    //             sum = __hadd(sum, __hmul(in[n * C * H * W + c * H * W + h * W + w],
                    //                                      weight[k * C * R * S + c * R * S + r * S + s]));
                    //         }
                    //         w += dilation;
                    //     }
                    // }
                }
            }
        }
        out[n * K * OH * OW + k * OH * OW + oh * OW + ow] = sum;
    }
}


void Conv2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out, cudaStream_t stream) {
    size_t N = in->shape[0];
    size_t C = in->shape[1];
    size_t H = in->shape[2];
    size_t W = in->shape[3];
    size_t K = weight->shape[0];
    size_t R = weight->shape[2];
    size_t S = weight->shape[3];
    size_t OH = out->shape[2];
    size_t OW = out->shape[3];

    const size_t stride = 1;
    const size_t pad = 1;
    const size_t dilation = 1;

    // Adjust block and grid dimensions
    dim3 block(16, 16);  // 256 threads per block
    dim3 grid(N, K, ((OH + block.x - 1) / block.x) * ((OW + block.y - 1) / block.y));

    Conv2d_kernel<<<grid, block, 0, stream>>>(in->d_buf, weight->d_buf, bias->d_buf, out->d_buf,
                                              N, C, H, W, K, R, S, OH, OW,
                                              stride, pad, dilation);
}

/* Tanh GPU kernel
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
__global__ void Tanh_kernel(half *inout, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        inout[idx] = __float2half(tanhf(__half2float(inout[idx])));
    }
}

/* Tanh using CUDA GPU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void Tanh(Tensor *inout, cudaStream_t stream) {
    size_t N = inout->num_elem();

    // Kernel launch configuration
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Launch kernel
    Tanh_kernel<<<numBlocks, blockSize, 0, stream>>>(inout->d_buf, N);

    // Check for errors
    CHECK_CUDA(cudaGetLastError());

    // Synchronize stream to ensure the kernel has completed
    CHECK_CUDA(cudaStreamSynchronize(stream));
}