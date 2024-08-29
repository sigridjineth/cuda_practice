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
__global__ void LinearKernel(half *in, half *w, half *b, half *out, size_t batch_size, size_t M, size_t N, size_t K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.z;

    if (m < M && n < N && b_idx < batch_size) {
        half sum = __float2half(0.0f);
        for (size_t k = 0; k < K; k++) {
            sum = __hadd(sum, __hmul(in[b_idx * M * K + m * K + k], w[n * K + k]));
        }
        out[b_idx * M * N + m * N + n] = __hadd(sum, b[n]);
    }
}

void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out, cudaStream_t stream, size_t batch_size) {
    size_t M = out->shape[0];
    size_t N = out->shape[1];
    size_t K = w->shape[1];

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y,
                 batch_size);
    LinearKernel<<<gridDim, blockDim, 0, stream>>>(in->d_buf, w->d_buf, b->d_buf, out->d_buf, batch_size, M, N, K);
}

/* Reshape
 * @param [in]   in: [batch_size, N, D]
 * @param [out] out: [batch_size, N, C, H, W]
 */
__global__ void ReshapeKernel(half *in, half *out, size_t batch_size,
                              size_t N, size_t D, size_t C, size_t H, size_t W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = blockIdx.y;

    if (b_idx < batch_size && idx < N * C * H * W) {
        size_t n = idx / (C * H * W);
        size_t chw = idx % (C * H * W);
        out[b_idx * N * C * H * W + n * C * H * W + chw] = in[b_idx * N * D + n * D + chw];
    }
}

void Reshape(Tensor *in, Tensor *out, cudaStream_t stream, size_t batch_size) {
    size_t N = in->shape[0];
    size_t D = in->shape[1];
    size_t C = out->shape[1];
    size_t H = out->shape[2];
    size_t W = out->shape[3];

    int totalThreads = N * C * H * W;
    int blockSize = 256;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;
    dim3 gridDim(numBlocks, batch_size);
    dim3 blockDim(blockSize);
    ReshapeKernel<<<gridDim, blockDim, 0, stream>>>(in->d_buf, out->d_buf, batch_size, N, D, C, H, W);
}

/* ConvTranspose2d
 * @param [in1]     in: [batch_size, N, C, H, W]
 * @param [in2] weight: [C, K, R, S]
 * @param [in3]   bias: [K]
 * @param [out]    out: [batch_size, N, K, OH, OW]
 */
__global__ void ConvTranspose2dKernel(const half* __restrict__ in,
                                      const half* __restrict__ weight,
                                      const half* __restrict__ bias,
                                      half* __restrict__ out,
                                      size_t batch_size, size_t N, size_t C, size_t H, size_t W,
                                      size_t K, size_t R, size_t S, size_t OH, size_t OW,
                                      int stride, int pad, int dilation) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;
    int b_idx = blockIdx.z;

    if (b_idx < batch_size && k < K && oh < OH && ow < OW) {
        float sum = 0.0f;

        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                for (int s = 0; s < S; ++s) {
                    int h = (oh + pad - r * dilation) / stride;
                    int w = (ow + pad - s * dilation) / stride;
                    if (h >= 0 && h < H && w >= 0 && w < W &&
                        (oh + pad - r * dilation) % stride == 0 &&
                        (ow + pad - s * dilation) % stride == 0) {
                        float in_val = __half2float(in[b_idx * N * C * H * W + c * H * W + h * W + w]);
                        float weight_val = __half2float(weight[c * K * R * S + k * R * S + r * S + s]);
                        sum += in_val * weight_val;
                    }
                }
            }
        }

        sum += __half2float(bias[k]);
        out[b_idx * N * K * OH * OW + k * OH * OW + oh * OW + ow] = __float2half(sum);
    }
}

void ConvTranspose2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out, cudaStream_t stream, size_t batch_size) {
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

    dim3 blockDim(8, 8, 8);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x,
                 (OH + blockDim.y - 1) / blockDim.y,
                 (OW + blockDim.z - 1) / blockDim.z);
    gridDim.z = batch_size;  // batch_size를 gridDim.z에 할당

    ConvTranspose2dKernel<<<gridDim, blockDim, 0, stream>>>(
            in->d_buf, weight->d_buf, bias->d_buf, out->d_buf,
            batch_size, N, C, H, W, K, R, S, OH, OW,
            stride, pad, dilation);
}

/* BatchNorm2d (track_running_stats=False)
 * @param [in1]     in: [batch_size, N, C, H, W]
 * @param [in2] weight: [C]
 * @param [in3]   bias: [C]
 * @param [out]    out: [batch_size, N, C, H, W]
 */
__global__ void BatchNorm2d_kernel(half *in, half *weight, half *bias, half *out,
                                   size_t batch_size, size_t N, size_t C, size_t H, size_t W) {
    const float eps = 1e-5f;
    size_t c = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;
    size_t HW = H * W;
    size_t b_idx = blockIdx.y;

    __shared__ float mean;
    __shared__ float var;

    // Step 1: Calculate mean
    float sum = 0.0f;
    for (size_t i = tid; i < HW; i += stride) {
        sum += __half2float(in[b_idx * N * C * HW + c * HW + i]);
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
        float diff = __half2float(in[b_idx * N * C * HW + c * HW + i]) - mean;
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
        float normalized = (__half2float(in[b_idx * N * C * HW + c * HW + i]) - mean) * invstd;
        out[b_idx * N * C * HW + c * HW + i] = __float2half(w * normalized + b);
    }
}

void BatchNorm2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out, cudaStream_t stream, size_t batch_size) {
    size_t N = in->shape[0];
    size_t C = in->shape[1];
    size_t H = in->shape[2];
    size_t W = in->shape[3];

    dim3 grid(C, batch_size);  // One block per channel
    dim3 block(256);  // Adjust this based on your GPU capabilities
    BatchNorm2d_kernel<<<grid, block, 0, stream>>>(in->d_buf, weight->d_buf, bias->d_buf, out->d_buf, batch_size, N, C, H, W);
}

/* LeakyReLU GPU kernel
 * @param [in & out] inout: [batch_size, N, C, H, W]
 */
__global__ void LeakyReLU_kernel(half *inout, size_t batch_size, size_t N, size_t C, size_t H, size_t W, half alpha) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t b_idx = blockIdx.y;

    if (b_idx < batch_size && idx < N * C * H * W) {
        size_t n = idx / (C * H * W);
        size_t chw = idx % (C * H * W);
        if (inout[b_idx * N * C * H * W + n * C * H * W + chw] < half(0)) {
            inout[b_idx * N * C * H * W + n * C * H * W + chw] *= alpha;
        }
    }
}

/* LeakyReLU using CUDA GPU
 * @param [in & out] inout: [batch_size, N, C, H, W]
 */
void LeakyReLU(Tensor *inout, cudaStream_t stream, size_t batch_size) {
    size_t N = inout->shape[0];
    size_t C = inout->shape[1];
    size_t H = inout->shape[2];
    size_t W = inout->shape[3];
    const half alpha = 0.01;

    dim3 blockDim(256);
    dim3 gridDim((N * C * H * W + blockDim.x - 1) / blockDim.x, batch_size);

    LeakyReLU_kernel<<<gridDim, blockDim, 0, stream>>>(inout->d_buf, batch_size, N, C, H, W, alpha);
}

/* Conv2d
 * @param [in1]     in: [batch_size, N, C, H, W]
 * @param [in2] weight: [K, C, R, S]
 * @param [in3]   bias: [K]
 * @param [out]    out: [batch_size, N, K, OH, OW]
 */
__global__ void Conv2d_kernel(half *in, half *weight, half *bias, half *out,
                              size_t batch_size, size_t N, size_t C, size_t H, size_t W,
                              size_t K, size_t R, size_t S,
                              size_t OH, size_t OW,
                              size_t stride, size_t pad, size_t dilation) {
    int n = blockIdx.x;
    int k = blockIdx.y;
    int oh_block = blockIdx.z / ((OW + blockDim.y - 1) / blockDim.y);
    int ow_block = blockIdx.z % ((OW + blockDim.y - 1) / blockDim.y);
    int oh = oh_block * blockDim.x + threadIdx.x;
    int ow = ow_block * blockDim.y + threadIdx.y;
    int b_idx = blockIdx.z;

    if (b_idx < batch_size && oh < OH && ow < OW) {
        half sum = bias[k];
        for (int c = 0; c < C; c++) {
            for (int r = 0; r < R; r++) {
                for (int s = 0; s < S; s++) {
                    int h = oh * stride - pad + r * dilation;
                    int w = ow * stride - pad + s * dilation;
                    if (h >= 0 && h < H && w >= 0 && w < W) {
                        sum = __hadd(sum, __hmul(in[b_idx * N * C * H * W + n * C * H * W + c * H * W + h * W + w],
                                                 weight[k * C * R * S + c * R * S + r * S + s]));
                    }
                }
            }
            out[b_idx * N * K * OH * OW + n * K * OH * OW + k * OH * OW + oh * OW + ow] = sum;
        }
    }
}

void Conv2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out, cudaStream_t stream, size_t batch_size) {
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

    dim3 block(16, 16);
    dim3 grid(N, K, ((OH + block.x - 1) / block.x) * ((OW + block.y - 1) / block.y));
    grid.z = batch_size;  // batch_size를 grid.z에 할당

    Conv2d_kernel<<<grid, block, 0, stream>>>(in->d_buf, weight->d_buf, bias->d_buf, out->d_buf,
                                              batch_size, N, C, H, W, K, R, S, OH, OW,
                                              stride, pad, dilation);
}

/* Tanh GPU kernel
 * @param [in & out] inout: [batch_size, N, C, H, W]
 */
    __global__ void Tanh_kernel(half *inout, size_t batch_size, size_t N, size_t C, size_t H, size_t W) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t b_idx = blockIdx.y;

        if (b_idx < batch_size && idx < N * C * H * W) {
            size_t n = idx / (C * H * W);
            size_t chw = idx % (C * H * W);
            inout[b_idx * N * C * H * W + n * C * H * W + chw] = __float2half(tanhf(__half2float(inout[b_idx * N * C * H * W + n * C * H * W + chw])));
        }
    }

/* Tanh using CUDA GPU
 * @param [in & out] inout: [batch_size, N, C, H, W]
 */
    void Tanh(Tensor *inout, cudaStream_t stream, size_t batch_size) {
        size_t N = inout->shape[0];
        size_t C = inout->shape[1];
        size_t H = inout->shape[2];
        size_t W = inout->shape[3];

        // Kernel launch configuration
        int blockSize = 256;
        int numBlocks = (N * C * H * W + blockSize - 1) / blockSize;

        // Launch kernel
        dim3 gridDim(numBlocks, batch_size);
        dim3 blockDim(blockSize);
        Tanh_kernel<<<gridDim, blockDim, 0, stream>>>(inout->d_buf, batch_size, N, C, H, W);

        // Check for errors
        CHECK_CUDA(cudaGetLastError());

        // Synchronize stream to ensure the kernel has completed
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }