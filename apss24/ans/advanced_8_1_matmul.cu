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

__global__ void matmul_kernel(float *A_T, float4 *B, float4 *C, int M, int N, int K) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i >= M || j * 4 >= N) return;
  float4 sum = make_float4(0, 0, 0, 0);
  for (int k = 0; k < K; ++k) {
    float a = A_T[k * M + i];
    float4 b = B[k * (N / 4) + j];
    sum = make_float4(sum.x + a * b.x, sum.y + a * b.y,
                      sum.z + a * b.z, sum.w + a * b.w);
  }
  C[i * (N / 4) + j] = sum;
}

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  CHECK_CUDA(
      cudaMemcpy(A_gpu, _A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(B_gpu, _B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  dim3 blockDim(32, 32);
  dim3 gridDim((M + 32 - 1) / 32, (N / 4 + 32 - 1) / 32);
  matmul_kernel<<<gridDim, blockDim>>>(A_gpu, (float4*)B_gpu, (float4*)C_gpu, M, N, K);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(
      cudaMemcpy(_C, C_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K) {
  CHECK_CUDA(cudaMalloc(&A_gpu, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B_gpu, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_gpu, M * N * sizeof(float)));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K) {
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
