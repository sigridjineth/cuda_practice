#include <cstdlib>

#include "convolution.cuh"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

void naive_cpu_im2col(float *_I, float *workspace, int N, int C, int H, int W,
                      int R, int S, int pad_h, int pad_w, int stride_h,
                      int stride_w, int dilation_h, int dilation_w) {
  float *I = _I;

  // Naive CPU im2col
  const int ON = N;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  for (int on = 0; on < ON; ++on) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        for (int c = 0; c < C; ++c) {
          for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
              const int n = on;
              const int h = oh * stride_h - pad_h + r * dilation_h;
              const int w = ow * stride_w - pad_w + s * dilation_w;

              if (h < 0 || h >= H || w < 0 || w >= W) continue;

              workspace[(((size_t)c * R * S) + (r * S) + s) * (ON * OH * OW) +
                        (on * OH * OW + oh * OW + ow)] =
                  I[(size_t)n * C * H * W + c * H * W + h * W + w];
            }
          }
        }
      }
    }
  }
}

void naive_cpu_matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        _C[(size_t)i * N + j] += _A[(size_t)i * K + k] * _B[(size_t)k * N + j];
      }
    }
  }
}

void reshape(float *_src, float *_dst, int N, int K, int OH, int OW) {
  size_t chunk = OH * OW;

  for (int on = 0; on < N; ++on) {
    for (int k = 0; k < K; ++k) {
      memcpy((void *) (_dst + (((size_t)on * K + k) * chunk)),
             (void *) (_src + (((size_t)k * N + on) * chunk)), chunk * sizeof(float));
    }
  }
}

void naive_cpu_convolution_im2col(float *_I, float *_F, float *_O, float *_BUF1,
                                  float *_BUF2, int N, int C, int H, int W,
                                  int K, int R, int S, int pad_h, int pad_w,
                                  int stride_h, int stride_w, int dilation_h,
                                  int dilation_w) {
  float *I = _I, *F = _F, *O = _O, *BUF1 = _BUF1, *BUF2 = _BUF2;

  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  naive_cpu_im2col(I, BUF1, N, C, H, W, R, S, pad_h, pad_w, stride_h, stride_w,
                   dilation_h, dilation_w);

  naive_cpu_matmul(F, BUF1, BUF2, K, N * OH * OW, C * R * S);

  reshape(BUF2, O, N, K, OH, OW);
}

void convolution(float *_I, float *_F, float *_O, float *_BUF1, float *_BUF2,
                 int N, int C, int H, int W, int K, int R, int S, int pad_h,
                 int pad_w, int stride_h, int stride_w, int dilation_h,
                 int dilation_w) {
  // Remove this line after you complete the convolution on GPU
  naive_cpu_convolution_im2col(_I, _F, _O, _BUF1, _BUF2, N, C, H, W, K, R, S,
                               pad_h, pad_w, stride_h, stride_w, dilation_h,
                               dilation_w);
}

void convolution_initialize(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w) {}

void convolution_cleanup(float *_I, float *_F, float *_O, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w) {}