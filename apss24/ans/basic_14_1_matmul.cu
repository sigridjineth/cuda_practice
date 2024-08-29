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

static __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                                     int K) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i >= M || j >= N) return;
  float sum = 0.0;
  for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[k * N + j];
  C[i * N + j] = sum;
}

// CUDA Unified Memory pointers
static float *A_unified, *B_unified, *C_unified;

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  // CPU memcpy instead of cudaMemcpy API
  memcpy(A_unified, _A, M * K * sizeof(float));
  memcpy(B_unified, _B, K * N * sizeof(float));

  dim3 blockDim(32, 32);
  dim3 gridDim((M + 32 - 1) / 32, (N + 32 - 1) / 32);
  matmul_kernel<<<gridDim, blockDim>>>(A_unified, B_unified, C_unified, M, N, K);
  memcpy(_C, C_unified, M * N * sizeof(float));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K) {
  CHECK_CUDA(cudaMallocManaged(&A_unified, M * K * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged(&B_unified, K * N * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged(&C_unified, M * N * sizeof(float)));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K) {
  CHECK_CUDA(cudaFree(A_unified));
  CHECK_CUDA(cudaFree(B_unified));
  CHECK_CUDA(cudaFree(C_unified));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
