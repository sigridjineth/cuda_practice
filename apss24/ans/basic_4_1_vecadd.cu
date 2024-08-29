#include <cstdio>

#include "vecadd.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

__global__ void vecadd_kernel(const int N, const float *a, const float *b, float *c) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tidx >= N) return;
  c[tidx] = a[tidx] + b[tidx];
}

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;

void vecadd(float *_A, float *_B, float *_C, int N) {
  // Upload A and B vector to GPU
  CHECK_CUDA(cudaMemcpy(A_gpu, _A, N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_gpu, _B, N * sizeof(float), cudaMemcpyHostToDevice));

  // Launch kernel on a GPU
  dim3 gridDim((N + 512 - 1) / 512);
  dim3 blockDim(512);
  vecadd_kernel<<<gridDim, blockDim>>>(N, A_gpu, B_gpu, C_gpu);

  // Download C vector from GPU
  CHECK_CUDA(cudaMemcpy(_C, C_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void vecadd_init(int N) {
  // Allocate device memory
  CHECK_CUDA(cudaMalloc(&A_gpu, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B_gpu, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_gpu, N * sizeof(float)));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void vecadd_cleanup(float *_A, float *_B, float *_C, int N) {
  // Free GPU memoryy
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
