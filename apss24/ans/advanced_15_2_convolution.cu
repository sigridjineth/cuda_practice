#include <cudnn.h>

#include <cstdio>
#include <cstdlib>

#include "convolution.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

#define CHECK_CUDNN(call)                                              \
  do {                                                                 \
    cudnnStatus_t status_ = call;                                      \
    if (status_ != CUDNN_STATUS_SUCCESS) {                             \
      fprintf(stderr, "CUDNN error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudnnGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

static cudnnHandle_t handle;
static cudnnTensorDescriptor_t input_desc;
static cudnnFilterDescriptor_t filter_desc;
static cudnnConvolutionDescriptor_t conv_desc;
static cudnnTensorDescriptor_t output_desc;
static int ON, OC, OH, OW;
static float *I_gpu, *F_gpu, *O_gpu, *workspace;
static cudnnConvolutionFwdAlgoPerf_t best_algo;

static const char *algo_to_string(cudnnConvolutionFwdAlgo_t algo);

void convolution(float *_I, float *_F, float *_O, int N, int C, int H, int W,
                 int K, int R, int S, int pad_h, int pad_w, int stride_h,
                 int stride_w, int dilation_h, int dilation_w) {
  CHECK_CUDA(cudaMemcpy(I_gpu, _I, N * C * H * W * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F_gpu, _F, K * C * R * S * sizeof(float),
                        cudaMemcpyHostToDevice));

  const float alpha = 1, beta = 0;
  CHECK_CUDNN(cudnnConvolutionForward(
      handle, &alpha, input_desc, I_gpu, filter_desc, F_gpu, conv_desc,
      best_algo.algo, workspace, best_algo.memory, &beta, output_desc, O_gpu));

  CHECK_CUDA(cudaMemcpy(_O, O_gpu, ON * OC * OH * OW * sizeof(float),
                        cudaMemcpyDeviceToHost));
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_initialize(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w) {
  CHECK_CUDNN(cudnnCreate(&handle));

  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, N, C, H, W));

  CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
  CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
                                         CUDNN_TENSOR_NCHW, K, C, R, S));

  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
      conv_desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
      conv_desc, input_desc, filter_desc, &ON, &OC, &OH, &OW));

  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, ON, OC, OH, OW));

  int max_algo_count;
  CHECK_CUDNN(
      cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &max_algo_count));

  int returned_algo_count;
  cudnnConvolutionFwdAlgoPerf_t algo_perfs[max_algo_count];
  CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
      handle, input_desc, filter_desc, conv_desc, output_desc, max_algo_count,
      &returned_algo_count, algo_perfs));

  for (int i = 0; i < returned_algo_count; ++i) {
    printf("Algorithm %d: name %s, time %f sec, memory %lu byte, status %s\n",
           i, algo_to_string(algo_perfs[i].algo), algo_perfs[i].time,
           algo_perfs[i].memory, cudnnGetErrorString(algo_perfs[i].status));
  }

  best_algo = algo_perfs[0];

  CHECK_CUDA(cudaMalloc(&I_gpu, N * C * H * W * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&F_gpu, K * C * R * S * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&O_gpu, ON * OC * OH * OW * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&workspace, algo_perfs[0].memory));
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_cleanup(float *_I, float *_F, float *_O, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w) {
  CHECK_CUDA(cudaFree(I_gpu));
  CHECK_CUDA(cudaFree(F_gpu));
  CHECK_CUDA(cudaFree(O_gpu));
  CHECK_CUDA(cudaFree(workspace));

  CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
  CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
  CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
  CHECK_CUDNN(cudnnDestroy(handle));
  CHECK_CUDA(cudaDeviceSynchronize());
}

const char *algo_to_string(cudnnConvolutionFwdAlgo_t algo) {
  switch (algo) {
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
      return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
      return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
      return "CUDNN_CONVOLUTION_FWD_ALGO_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
      return "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
      return "CUDNN_CONVOLUTION_FWD_ALGO_FFT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
      return "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
      return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
      return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED";
    case CUDNN_CONVOLUTION_FWD_ALGO_COUNT:
      return "CUDNN_CONVOLUTION_FWD_ALGO_COUNT";
    default: return "<unknown algorithm>";
  }
}