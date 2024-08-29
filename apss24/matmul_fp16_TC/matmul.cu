#include <cstdio>
#include <mma.h>

#include "matmul.h"

using namespace nvcuda;

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;

void naive_cpu_matmul(half *_A, half *_B, float *_C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        _C[i * N + j] += (float)_A[i * K + k] * (float)_B[k * N + j];
      }
    }
  }
}

void matmul(half *_A, half *_B, float *_C, int M, int N, int K) {
  // Remove this line after you complete the matmul on GPU
  naive_cpu_matmul(_A, _B, _C, M, N, K);

  // (TODO) Upload A and B matrix to GPU

  // (TODO) Launch kernel on a GPU

  // (TODO) Download C matrix from GPU

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K) {
  // (TODO) Allocate device memory

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cleanup(half *_A, half *_B, float *_C, int M, int N, int K) {
  // (TODO) Do any post-matmul cleanup work here.

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
