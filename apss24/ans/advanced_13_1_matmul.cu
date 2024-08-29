#include <cstdio>

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
#define EVENTS_PER_GPU 1  // Increase as needed

static int Mbegin[NGPU], Mend[NGPU];
static int ngpu;
static cudaStream_t streams[NGPU];
static cudaEvent_t events[NGPU][EVENTS_PER_GPU];
static float *A_gpu[NGPU], *B_gpu[NGPU], *C_gpu[NGPU];

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(A_gpu[i], &_A[Mbegin[i] * K],
                               (Mend[i] - Mbegin[i]) * K * sizeof(float),
                               cudaMemcpyHostToDevice, streams[i]));
    CHECK_CUDA(cudaMemcpyAsync(B_gpu[i], _B, K * N * sizeof(float),
                               cudaMemcpyHostToDevice, streams[i]));
  }

  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    dim3 blockDim(16, 16);
    dim3 gridDim((N + 16 - 1) / 16, (Mend[i] - Mbegin[i] + 16 - 1) / 16);
    matmul_kernel<<<gridDim, blockDim, 0, streams[i]>>>(
        A_gpu[i], B_gpu[i], C_gpu[i], Mend[i] - Mbegin[i], N, K);
    CHECK_CUDA(cudaGetLastError());
  }

  for (int i = 0; i < ngpu; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(&_C[Mbegin[i] * N], C_gpu[i],
                               (Mend[i] - Mbegin[i]) * N * sizeof(float),
                               cudaMemcpyDeviceToHost, streams[i]));
  }

  for (int i = 0; i < ngpu; i++) {
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i]);
  }
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
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
    for (int j = 0; j < EVENTS_PER_GPU; j++) {
      CHECK_CUDA(cudaEventCreate(&events[i][j]));
    }
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
    CHECK_CUDA(cudaStreamDestroy(streams[i]));
    for (int j = 0; j < EVENTS_PER_GPU; j++) {
      CHECK_CUDA(cudaEventDestroy(events[i][j]));
    }
  }
}