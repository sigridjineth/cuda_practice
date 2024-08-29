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
 * GPU 병렬화: 행렬-벡터 곱셈 알고리즘을 사용하여 병렬화합니다.
 * half 정밀도 활용: GPU의 native half 타입을 사용하여 연산 속도를 향상시킵니다.
 */
__global__ void LinearKernel(half *in, half *w, half *b, half *out,
                             size_t M, size_t N, size_t K) {
    __shared__ half weights[16][16];
    __shared__ half input[16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    half sum = __float2half(0.0f);
    for (int i = 0; i < K; i += 16) {
        int input_idx = (by * 16 + ty) * K + i + tx;
        int weight_idx = (bx * 16 + tx) * K + i + ty;

        if (input_idx < M * K) {
            input[tx] = in[input_idx];
        } else {
            input[tx] = __float2half(0.0f);
        }

        if (weight_idx < N * K) {
            weights[tx][ty] = w[weight_idx];
        }

        __syncthreads();

        for (int j = 0; j < 16; j++) {
            sum = __hadd(sum, __hmul(input[j], weights[tx][j]));
        }

        __syncthreads();
    }

    int output_idx = (by * 16 + ty) * N + bx * 16 + tx;
    if (output_idx < M * N) {
        out[output_idx] = __hadd(sum, b[bx * 16 + tx]);
    }
}

void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out, cudaStream_t stream) {
    size_t M = out->shape[0];
    size_t N = out->shape[1];
    size_t K = w->shape[1];

    dim3 blockDim(16, 16);
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
    extern __shared__ half shmem[];
    half *weight_tile = shmem;
    half *input_tile = &shmem[K * R * S];

    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;

    if (k >= K || oh >= OH || ow >= OW) return;

    float sum = 0.0f;

    // Load weight tile into shared memory
    for (int i = 0; i < K * R * S; i += blockDim.x) {
        int weight_idx = k * R * S + i + threadIdx.x;
        if (weight_idx < K * R * S) {
            weight_tile[i + threadIdx.x] = weight[weight_idx];
        }
    }

    // Load input tile into shared memory
    for (int i = 0; i < C; i += blockDim.y) {
        for (int j = 0; j < R * S; j += blockDim.x) {
            int input_idx = (i + threadIdx.y) * H * W + (oh * stride - pad + (j / S) * dilation) * W + ow * stride - pad + (j % S) * dilation;
            if (i + threadIdx.y < C && input_idx >= 0 && input_idx < H * W) {
                input_tile[(i + threadIdx.y) * R * S + j + threadIdx.x] = in[(i + threadIdx.y) * H * W + input_idx];
            } else {
                input_tile[(i + threadIdx.y) * R * S + j + threadIdx.x] = __float2half(0.0f);
            }
        }
    }

    __syncthreads();

    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                float in_val = __half2float(input_tile[c * R * S + r * S + s]);
                float weight_val = __half2float(weight_tile[k * R * S + r * S + s]);
                sum += in_val * weight_val;
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

    int shared_mem_size = (K * R * S + C * R * S) * sizeof(half);

    dim3 blockDim(8, 4, 4);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x,
                 (OH + blockDim.y - 1) / blockDim.y,
                 (OW + blockDim.z - 1) / blockDim.z);

    ConvTranspose2dKernel<<<gridDim, blockDim, shared_mem_size, stream>>>(
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

    __shared__ float mean_sum, var_sum;
    __shared__ float mean, var;

    float thread_sum = 0.0f;
    float thread_var_sum = 0.0f;

    for (size_t i = tid; i < HW; i += stride) {
        size_t idx = c * HW + i;
        if (idx < C * HW) {
            float val = __half2float(in[idx]);
            thread_sum += val;
            thread_var_sum += val * val;
        }
    }

    mean_sum = warp_reduce(thread_sum);
    var_sum = warp_reduce(thread_var_sum);

    if (tid == 0) {
        mean = mean_sum / (float)HW;
        var = var_sum / (float)HW - mean * mean;
    }

    __syncthreads();

    float w = __half2float(weight[c]);
    float b = __half2float(bias[c]);
    float invstd = rsqrtf(var + eps);

    for (size_t i = tid; i < HW; i += stride) {
        size_t idx = c * HW + i;
        if (idx < C * HW) {
            float normalized = (__half2float(in[idx]) - mean) * invstd;
            out[idx] = __float2half(w * normalized + b);
        }
    }
}

__inline__ __device__ float warp_reduce(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

void BatchNorm2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out, cudaStream_t stream) {
    size_t N = in->shape[0];
    size_t C = in->shape[1];
    size_t H = in->shape[2];
    size_t W = in->shape[3];

    dim3 grid(C);
    dim3 block(256);
    BatchNorm2d_kernel<<<grid, block, 0, stream>>>(in->d_buf, weight->d_buf, bias->d_buf, out->d_buf, N, C, H, W);
}

/* LeakyReLU GPU kernel
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
__global__ void LeakyReLU_kernel(half *inout, size_t N, half alpha) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < N; i += stride) {
        inout[i] = (inout[i] < half(0)) ? __hmul(inout[i], alpha) : inout[i];
    }
}

/* LeakyReLU using CUDA GPU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void LeakyReLU(Tensor *inout, cudaStream_t stream) {
    size_t N = inout->num_elem();
    const half alpha = 0.01;

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    LeakyReLU_kernel<<<numBlocks, blockSize, 0, stream>>>(inout->d_buf, N, alpha);
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

    if (oh < OH && ow < OW && n < N && k < K) {
        half sum = bias[k];
        for (int c = 0; c < C; c++) {
            for (int r = 0; r < R; r++) {
                int h = oh * stride - pad + r * dilation;
                int w = ow * stride - pad;
                if (h >= 0 && h < H) {
#pragma unroll
                    for (int s = 0; s < S; s++) {
                        int input_idx = n * C * H * W + c * H * W + h * W + w;
                        if (w >= 0 && w < W && input_idx < n * C * H * W) {
                            sum = __hadd(sum, __hmul(in[input_idx],
                                                     weight[k * C * R * S + c * R * S + r * S + s]));
                        }
                        w += dilation;
                    }
                }
            }
        }
        int output_idx = n * K * OH * OW + k * OH * OW + oh * OW + ow;
        if (output_idx < n * K * OH * OW) {
            out[output_idx] = sum;
        }
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

    dim3 block(16, 16);
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

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    Tanh_kernel<<<numBlocks, blockSize, 0, stream>>>(inout->d_buf, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
}