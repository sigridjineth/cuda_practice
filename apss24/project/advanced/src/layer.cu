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
void ConvTranspose2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
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

#pragma omp parallel for
  for (size_t oc = 0; oc < K; ++oc) {
    for (size_t oh = 0; oh < OH; ++oh) {
      for (size_t ow = 0; ow < OW; ++ow) {
        half_cpu o = bias->buf[oc];
        for (size_t c = 0; c < C; ++c) {
          for (size_t r = 0; r < R; ++r) {
            for (size_t s = 0; s < S; ++s) {
              if ((oh - (r * dilation - pad)) % stride != 0) continue;
              if ((ow - (s * dilation - pad)) % stride != 0) continue;
              size_t h = (oh - (r * dilation - pad)) / stride;
              size_t w = (ow - (s * dilation - pad)) / stride;
              if (h >= H || w >= W) continue;
              o += in->buf[c * H * W + h * W + w] * 
                weight->buf[c * K * R * S + oc * R * S + r * S + s];
            }
          }
        }
        out->buf[oc * OH * OW + oh * OW + ow] = o;
      }
    }
  }
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
void BatchNorm2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out) {
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];

  const float eps = 1e-5f;

  for (size_t c = 0; c < C; c++) {
    // 1. Caculate mean for each channel
    float mean = 0.0f;
    float var = 0.0f;
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        half_cpu val = in->buf[c * H * W + h * W + w];
        mean += static_cast<float>(val); /* Cast to float */
      }
    }
    mean /= static_cast<float>(H * W);

    // 2. Caculate variance for each channel
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        half_cpu val = in->buf[c * H * W + h * W + w];
        var += (static_cast<float>(val) - mean) * 
          (static_cast<float>(val) - mean); /* Cast to float */
      }
    }
    var /= static_cast<float>(H * W);

    // 3. Normalize with the calculated mean and variance
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        out->buf[c * H * W + h * W + w] =
          weight->buf[c] * 
          (in->buf[c * H * W + h * W + w] - 
          half_cpu(mean)) / /* Cast to half */
          half_cpu(sqrt(var + eps)) + /* Cast to half */
          bias->buf[c];
      }
    }
  }
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

/* Tanh 
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void Tanh(Tensor *inout) {
  size_t N = inout->num_elem();

  for (size_t i = 0; i < N; i++) {
    inout->buf[i] = tanh(inout->buf[i]);
  }
}

