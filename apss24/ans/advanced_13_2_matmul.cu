#include <cstdio>
#include <thread>

#include "matmul.h"

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

static __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                                     int K) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  if (i >= M || j >= N) return;
  float sum = 0.0;
  for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[k * N + j];
  C[i * N + j] = sum;
}

#define NGPU 4

static int Mbegin[NGPU], Mend[NGPU];
static int ngpu;
static float *A_gpu[NGPU], *B_gpu[NGPU], *C_gpu[NGPU];

void matmul_thread(float *_A, float *_B, float *_C, int M, int N, int K,
                   int gpu_id) {
  CHECK_CUDA(cudaSetDevice(gpu_id));
  int num_rows = Mend[gpu_id] - Mbegin[gpu_id];

  CHECK_CUDA(cudaMemcpy(A_gpu[gpu_id], &_A[Mbegin[gpu_id] * K],
                        num_rows * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_gpu[gpu_id], _B, K * N * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 blockDim(16, 16);
  dim3 gridDim((N + 16 - 1) / 16, (num_rows + 16 - 1) / 16);
  matmul_kernel<<<gridDim, blockDim>>>(
      A_gpu[gpu_id], B_gpu[gpu_id], C_gpu[gpu_id], num_rows, N, K);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(&_C[Mbegin[gpu_id] * N], C_gpu[gpu_id],
                        num_rows * N * sizeof(float), cudaMemcpyDeviceToHost));
}

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  std::thread threads[NGPU];
  for (int i = 0; i < NGPU; i++) {
    threads[i] = std::thread(matmul_thread, _A, _B, _C, M, N, K, i);
  }
  for (int i = 0; i < NGPU; i++) { threads[i].join(); }
}

void matmul_init(int M, int N, int K) {
  ngpu = 4;

  for (int i = 0; i < ngpu; i++) {
    Mbegin[i] = M / ngpu * i;
    Mend[i] = M / ngpu * (i + 1);
    if (i == ngpu - 1) Mend[i] = M;
  }

  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(
        cudaMalloc(&A_gpu[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B_gpu[i], K * N * sizeof(float)));
    CHECK_CUDA(
        cudaMalloc(&C_gpu[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
  }
}

void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K) {
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(A_gpu[i]));
    CHECK_CUDA(cudaFree(B_gpu[i]));
    CHECK_CUDA(cudaFree(C_gpu[i]));
  }
}