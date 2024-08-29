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

void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
    size_t M = out->shape[0];
    size_t N = out->shape[1];
    size_t K = w->shape[1];

    half *d_in, *d_w, *d_b, *d_out;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_in, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_w, N * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_out, M * N * sizeof(half)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_in, in->buf, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, w->buf, N * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b->buf, N * sizeof(half), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    LinearKernel<<<gridDim, blockDim>>>(d_in, d_w, d_b, d_out, M, N, K);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(out->buf, d_out, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_out));
}

/* Reshape 
 * @param [in]   in: [N, D]
 * @param [out] out: [N, C, H, W]
 * 'N' is the number of input tensors.
 * 'D' is the dimension of the input tensor.
 * 'C' is the number of channels.
 * 'H' is the height of the output tensor.
 * 'W' is the width of the output tensor.
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

void Reshape(Tensor *in, Tensor *out) {
    size_t N = in->shape[0];
    size_t D = in->shape[1];
    size_t C = out->shape[1];
    size_t H = out->shape[2];
    size_t W = out->shape[3];

    half *d_in, *d_out;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_in, N * D * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_out, N * C * H * W * sizeof(half)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_in, in->buf, N * D * sizeof(half), cudaMemcpyHostToDevice));

    // Launch kernel
    int totalThreads = N * C * H * W;
    int blockSize = 256;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;
    ReshapeKernel<<<numBlocks, blockSize>>>(d_in, d_out, N, D, C, H, W);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(out->buf, d_out, N * C * H * W * sizeof(half), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

__global__ void ConvTranspose2dBatchNormLeakyReLUKernel(
        const half* in, const half* weight, const half* bias,
        const half* bn_weight, const half* bn_bias, half* out,
        int N, int C, int H, int W, int K, int R, int S, int OH, int OW) {

    const float eps = 1e-5f;
    const float alpha = 0.01f; // LeakyReLU alpha

    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= K || oh >= OH || ow >= OW) return;

    const int stride = 2;
    const int pad = 1;
    const int dilation = 1;

    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int h = (oh + pad - r * dilation) / stride;
                int w = (ow + pad - s * dilation) / stride;
                if (h < 0 || h >= H || w < 0 || w >= W) continue;
                if ((oh + pad - r * dilation) % stride != 0) continue;
                if ((ow + pad - s * dilation) % stride != 0) continue;

                float in_val = __half2float(in[c * H * W + h * W + w]);
                float weight_val = __half2float(weight[k * C * R * S + c * R * S + r * S + s]);
                sum += in_val * weight_val;
            }
        }
    }

    // Add bias
    sum += __half2float(bias[k]);

    // BatchNorm2d
    float bn_weight_val = __half2float(bn_weight[k]);
    float bn_bias_val = __half2float(bn_bias[k]);

    // Note: For simplicity, we're using a global mean and variance of 0 and 1.
    // In a real scenario, you'd compute these values over the batch.
    float normalized = (sum - 0.0f) / sqrtf(1.0f + eps);
    float bn_out = bn_weight_val * normalized + bn_bias_val;

    // LeakyReLU
    float leaky_out = (bn_out > 0.0f) ? bn_out : alpha * bn_out;

    out[k * OH * OW + oh * OW + ow] = __float2half(leaky_out);
}

void ConvTranspose2dBatchNormLeakyReLU(Tensor *in, Tensor *weight, Tensor *bias,
                                       Tensor *bn_weight, Tensor *bn_bias, Tensor *out) {
    // ConvTranspose2d, BatchNorm2d, LeakyReLU를 하나의 커널로 융합
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((out->shape[3] + blockDim.x - 1) / blockDim.x,
                 (out->shape[2] + blockDim.y - 1) / blockDim.y,
                 (out->shape[1] + blockDim.z - 1) / blockDim.z);
    ConvTranspose2dBatchNormLeakyReLUKernel<<<gridDim, blockDim>>>(
            (const half*)in->buf, (const half*)weight->buf, (const half*)bias->buf,
            (const half*)bn_weight->buf, (const half*)bn_bias->buf, (half*)out->buf,
            in->shape[0], in->shape[1], in->shape[2], in->shape[3],
            out->shape[1], weight->shape[2], weight->shape[3],
            out->shape[2], out->shape[3]);
}

/* ConvTranspose2d
 * @param [in1]     in: [N, C, H, W]
 * @param [in2] weight: [C, K, R, S]
 * @param [in3]   bias: [K]
 * @param [out]    out: [N, K, OH, OW]
 *    
 *    OH = (H - 1) * stride - 2 * pad + dilation * (R - 1) + output_pad + 1
 *    OW = (W - 1) * stride - 2 * pad + dilation * (S - 1) + output_pad + 1
 *    In this model, R = S = 3, stride = 2, pad = 1, dilation = 1, output_pad = 1
 *
 * 'N' is the number of input tensors.
 * 'C' is the number of input channels.
 * 'H' is the height of the input tensor.
 * 'W' is the width of the input tensor.
 * 'K' is the number of output channels.
 * 'R' is the height of the filter.
 * 'S' is the width of the filter.
 * 'OH' is the height of the output tensor.
 * 'OW' is the width of the output tensor.
 */
__global__ void ConvTranspose2dKernel(const half* in, const half* weight, const half* bias,
                                      half* out, int N, int C, int H, int W,
                                      int K, int R, int S, int OH, int OW,
                                      int stride, int pad, int dilation) {
    int oc = blockIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.z + threadIdx.z;

    if (oh >= OH || ow >= OW) return;

    half sum = __float2half(0.0f);
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int h = (oh + pad - r * dilation) / stride;
                int w = (ow + pad - s * dilation) / stride;
                if (h < 0 || h >= H || w < 0 || w >= W) continue;
                if ((oh + pad - r * dilation) % stride != 0) continue;
                if ((ow + pad - s * dilation) % stride != 0) continue;

                half in_val = in[c * H * W + h * W + w];
                half weight_val = weight[c * K * R * S + oc * R * S + r * S + s];
                sum = __hadd(sum, __hmul(in_val, weight_val));
            }
        }
    }
    out[oc * OH * OW + oh * OW + ow] = __hadd(sum, bias[oc]);
}

void ConvTranspose2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
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

    half *d_in, *d_weight, *d_bias, *d_out;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_in, N * C * H * W * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_weight, C * K * R * S * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_bias, K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_out, N * K * OH * OW * sizeof(half)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_in, in->buf, N * C * H * W * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, weight->buf, C * K * R * S * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, bias->buf, K * sizeof(half), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockDim(1, 16, 16);
    dim3 gridDim(K, (OH + blockDim.y - 1) / blockDim.y, (OW + blockDim.z - 1) / blockDim.z);
    ConvTranspose2dKernel<<<gridDim, blockDim>>>(d_in, d_weight, d_bias, d_out,
                                                 N, C, H, W, K, R, S, OH, OW,
                                                 stride, pad, dilation);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(out->buf, d_out, N * K * OH * OW * sizeof(half), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_out));
}

/* BatchNorm2d (track_running_stats=False)
 * @param [in1]     in: [N, C, H, W]
 * @param [in2] weight: [C]
 * @param [in3]   bias: [C]
 * @param [out]    out: [N, C, H, W]  
 * 
 *    out = weight * (in - mean) / sqrt(var + 1e-5) + bias 
 * 
 * 'N' is the number of input tensors.
 * 'C' is the number of channels.
 * 'H' is the height of the input tensor.
 * 'W' is the width of the input tensor.
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

void BatchNorm2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
    size_t N = in->shape[0];
    size_t C = in->shape[1];
    size_t H = in->shape[2];
    size_t W = in->shape[3];

    half *d_in, *d_weight, *d_bias, *d_out;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_in, N * C * H * W * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_weight, C * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_bias, C * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_out, N * C * H * W * sizeof(half)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_in, in->buf, N * C * H * W * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, weight->buf, C * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, bias->buf, C * sizeof(half), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 grid(C);  // One block per channel
    dim3 block(256);  // Adjust this based on your GPU capabilities
    BatchNorm2d_kernel<<<grid, block>>>(d_in, d_weight, d_bias, d_out, N, C, H, W);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(out->buf, d_out, N * C * H * W * sizeof(half), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_out));
}

/* LeakyReLU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void LeakyReLU(Tensor *inout) {
  size_t N = inout->num_elem();

  const half_cpu alpha = 0.01_h;

  for (size_t i = 0; i < N; i++) {
    if (inout->buf[i] < 0) { inout->buf[i] *= alpha; }
  }
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
void LeakyReLU_cuda(Tensor *inout) {
  size_t N = inout->num_elem();

  const half alpha = 0.01;
  
  half *d_inout;

  CHECK_CUDA(cudaMalloc(&d_inout, N * sizeof(half)));
  CHECK_CUDA(cudaMemcpy(d_inout, inout->buf, N * sizeof(half), cudaMemcpyHostToDevice));

  LeakyReLU_kernel<<<(N + 255) / 256, 256>>>(d_inout, N, alpha);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(inout->buf, d_inout, N * sizeof(half), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_inout));
}

/* Conv2d
 * @param [in1]     in: [N, C, H, W]
 * @param [in2] weight: [K, C, R, S]
 * @param [in3]   bias: [K]
 * @param [out]    out: [N, K, OH, OW]
 *
 *   OH = (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1
 *   OW = (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1
 *   In this model, R = S = 3, stride = 1, pad = 1, dilation = 1
 *
 * 'N' is the number of input tensors.
 * 'C' is the number of input channels.
 * 'H' is the height of the input tensor.
 * 'W' is the width of the input tensor.
 * 'K' is the number of output channels.
 * 'R' is the height of the filter.
 * 'S' is the width of the filter.
 * 'OH' is the height of the output tensor.
 * 'OW' is the width of the output tensor.
 */
void Conv2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
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

  for (size_t n = 0; n < N; n++) {
    for (size_t oc = 0; oc < K; oc++) {
      for (size_t oh = 0; oh < OH; oh++) {
        for (size_t ow = 0; ow < OW; ow++) {
          half_cpu o = bias->buf[oc];
          for (size_t c = 0; c < C; c++) {
            for (size_t r = 0; r < R; r++) {
              for (size_t s = 0; s < S; s++) {
                size_t h = oh * stride - pad + r * dilation;
                size_t w = ow * stride - pad + s * dilation;
                if (h >= H || w >= W) continue;
                o += in->buf[n * C * H * W + c * H * W + h * W + w] *
                  weight->buf[oc * C * R * S + c * R * S + r * S + s];
              }
            }
          }
          out->buf[n * K * OH * OW + oc * OH * OW + oh * OW + ow] = o;
        }
      }
    }
  }
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
void Tanh(Tensor *inout) {
    size_t N = inout->num_elem();

    half *d_inout;

    CHECK_CUDA(cudaMalloc(&d_inout, N * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(d_inout, inout->buf, N * sizeof(half), cudaMemcpyHostToDevice));

    Tanh_kernel<<<(N + 255) / 256, 256>>>(d_inout, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(inout->buf, d_inout, N * sizeof(half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_inout));
}

