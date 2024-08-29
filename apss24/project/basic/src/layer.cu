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
 * @param [in1]  in: [M, K]
 * @param [in2]   w: [N, K]
 * @param [in3]   b: [N]
 * @param [out] out: [M, N]
 */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t M = out->shape[0];
  size_t N = out->shape[1];
  size_t K = w->shape[1];

  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      out->buf[m * N + n] = 0;
      for (size_t k = 0; k < K; k++) {
        out->buf[m * N + n] += in->buf[m * K + k] * w->buf[n * K + k];
      }
      out->buf[m * N + n] += b->buf[n];
    }
  }
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
void Reshape(Tensor *in, Tensor *out) {
  size_t N = in->shape[0];
  size_t D = in->shape[1];
  size_t C = out->shape[1];
  size_t H = out->shape[2];
  size_t W = out->shape[3];

  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; h++) {
        for (size_t w = 0; w < W; w++) {
          out->buf[n * C * H * W + c * H * W + h * W + w] =
              in->buf[n * D + c * H * W + h * W + w];
        }
      }
    }
  }
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

  for (size_t oc = 0; oc < K; ++oc) {
    for (size_t oh = 0; oh < OH; ++oh) {
      for (size_t ow = 0; ow < OW; ++ow) {
        float o = bias->buf[oc];
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
        float val = in->buf[c * H * W + h * W + w];
        mean += val;
      }
    }
    mean /= (H * W);

    // 2. Caculate variance for each channel
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        float val = in->buf[c * H * W + h * W + w];
        var += (val - mean) * (val - mean);
      }
    }
    var /= (H * W);

    // 3. Normalize with the calculated mean and variance
    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        out->buf[c * H * W + h * W + w] =
          weight->buf[c] * 
          (in->buf[c * H * W + h * W + w] - mean) /
          sqrt(var + eps) +
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

  const float alpha = 0.01;

  for (size_t i = 0; i < N; i++) {
    if (inout->buf[i] < 0) { inout->buf[i] *= alpha; }
  }
}

/* LeakyReLU GPU kernel
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
__global__ void LeakyReLU_kernel(float *inout, size_t N, float alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) { 
    if (inout[idx] < 0) { inout[idx] *= alpha; }
  }
}  

/* LeakyReLU using CUDA GPU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void LeakyReLU_cuda(Tensor *inout) {
  size_t N = inout->num_elem();

  const float alpha = 0.01;

  float *d_inout;

  CHECK_CUDA(cudaMalloc(&d_inout, N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_inout, inout->buf, N * sizeof(float), cudaMemcpyHostToDevice));

  LeakyReLU_kernel<<<(N + 255) / 256, 256>>>(d_inout, N, alpha);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(inout->buf, d_inout, N * sizeof(float), cudaMemcpyDeviceToHost));
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
          float o = bias->buf[oc];
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
