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

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    LinearKernel<<<gridDim, blockDim, 0, stream>>>(in->d_buf, w->d_buf, b->d_buf, out->d_buf, M, N, K);
}

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

__global__ void ConvTranspose2dBatchNormLeakyReLUKernel(
        const half* in, const half* weight, const half* bias,
        const half* bn_weight, const half* bn_bias,
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
    sum = __hadd(sum, bias[oc]);

    // BatchNorm2d
    float bn_scale = __half2float(bn_weight[oc]);
    float bn_shift = __half2float(bn_bias[oc]);
    float eps = 1e-5f;
    float normalized = __half2float(sum) * bn_scale + bn_shift;

    // LeakyReLU
    if (normalized < 0) normalized *= 0.01f;

    out[oc * OH * OW + oh * OW + ow] = __float2half(normalized);
}

void ConvTranspose2dBatchNormLeakyReLU(Tensor *in, Tensor *weight, Tensor *bias,
                                       Tensor *bn_weight, Tensor *bn_bias,
                                       Tensor *out, cudaStream_t stream) {
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

    dim3 blockDim(1, 16, 16);
    dim3 gridDim(K, (OH + blockDim.y - 1) / blockDim.y, (OW + blockDim.z - 1) / blockDim.z);
    ConvTranspose2dBatchNormLeakyReLUKernel<<<gridDim, blockDim, 0, stream>>>(
            in->d_buf, weight->d_buf, bias->d_buf,
            bn_weight->d_buf, bn_bias->d_buf,
            out->d_buf, N, C, H, W, K, R, S, OH, OW,
            stride, pad, dilation);
}

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
                for (int s = 0; s < S; s++) {
                    int h = oh * stride - pad + r * dilation;
                    int w = ow * stride - pad + s * dilation;
                    if (h >= 0 && h < H && w >= 0 && w < W) {
                        sum = __hadd(sum, __hmul(in[n * C * H * W + c * H * W + h * W + w],
                                                 weight[k * C * R * S + c * R * S + r * S + s]));
                    }
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

    dim3 block(16, 16);
    dim3 grid(N, K, ((OH + block.x - 1) / block.x) * ((OW + block.y - 1) / block.y));

    Conv2d_kernel<<<grid, block, 0, stream>>>(in->d_buf, weight->d_buf, bias->d_buf, out->d_buf,
                                              N, C, H, W, K, R, S, OH, OW,
                                              stride, pad, dilation);
}

__global__ void Tanh_kernel(half *inout, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        inout[idx] = __float2half(tanhf(__half2float(inout[idx])));
    }
}

void Tanh(Tensor *inout, cudaStream_t stream) {
    size_t N = inout->num_elem();
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    Tanh_kernel<<<numBlocks, blockSize, 0, stream>>>(inout->d_buf, N);
}