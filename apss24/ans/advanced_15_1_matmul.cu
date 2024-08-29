#include <cublas_v2.h>
#include <cstdio>

#include "matmul.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

  #define CHECK_CUBLAS(call)                                                   \
  do {                                                                       \
    cublasStatus_t status_ = call;                                           \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                  \
      fprintf(stderr, "CUBLAS error (%s:%d): %s, %s\n", __FILE__, __LINE__,  \
              cublasGetStatusName(status_), cublasGetStatusString(status_)); \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

static float *A_gpu, *B_gpu, *C_gpu;
static cublasHandle_t handle;

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Matrix A is stored at A, with size (row=M, col=K), as row-major
  // It can be treated as A^T, with size (row=K, col=M), as column-major
  // We now copy A^T from A to A_gpu.
  CHECK_CUBLAS(cublasSetMatrix(K, M, sizeof(float), _A, K, A_gpu, K));

  // Matrix B is stored at B, with size (row=K, col=N), as row-major
  // It can be treated as B^T, with size (row=N, col=K), as column-major
  // We now copy B^T from B to B_gpu
  CHECK_CUBLAS(cublasSetMatrix(N, K, sizeof(float), _B, N, B_gpu, N));

  // Now we have A^T at A_gpu, B^T at B_gpu.
  // If we store C^T at C_gpu, it would be suitable to use cublasGetMatrix.
  // As C = A * B, C^T = B^T * A^T
  // So we multiply B^T and A^T with gemm
  // where m=N, n=M, k=K
  const float one = 1, zero = 0;
  CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one,
                           B_gpu, N, A_gpu, K, &zero, C_gpu, N));

  // Now we have C^T stored at C_gpu, with size (row=N, col=M), as column-major
  // After copying to C, it can be treated as C, with size (row=M, col=N), as
  // row-major
  CHECK_CUBLAS(cublasGetMatrix(N, M, sizeof(float), C_gpu, N, _C, N));
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K) {
  CHECK_CUDA(cudaMalloc((void **) &A_gpu, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **) &B_gpu, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **) &C_gpu, M * N * sizeof(float)));
  CHECK_CUBLAS(cublasCreate(&handle));
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K) {
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));
  CHECK_CUBLAS(cublasDestroy(handle));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
