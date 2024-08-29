#include <cstdio>

#include "transpose.h"

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
static float *A_gpu, *B_gpu;

void naive_cpu_transpose(float *A, float *B, int M, int N) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      B[i * M + j] = A[j * N + i];
    }
  }
}

// A: M x N matrix, B: N x M matrix
void transpose(float *_A, float *_B, int M, int N) {
  // Remove this line after you complete the transpose on GPU
  naive_cpu_transpose(_A, _B, M, N);

  // (TODO) Run transpose on GPU
  // You can memcpy data in initialize/cleanup functions.

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void transpose_init(float *_A, float *_B, int M, int N) {
  // (TODO) Allocate device memory

  // (TODO) Upload A matrix to GPU

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void transpose_cleanup(float *_A, float *_B, int M, int N) {
  // (TODO) Download B matrix from GPU

  // (TODO) Do any post-transpose cleanup work here.

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
