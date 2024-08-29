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

__global__ void vecadd_kernel(const int N, const float *a, const float *b,
                              float *c) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tidx >= N) return;
  c[tidx] = a[tidx] + b[tidx];
}

#define BLOCKS 8
static int Nbegin[BLOCKS], Nend[BLOCKS];
static cudaStream_t upload_stream, download_stream, calc_stream;
static cudaEvent_t upload_events[BLOCKS], calc_events[BLOCKS];

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;

void vecadd(float *_A, float *_B, float *_C, int N) {
  // Upload A and B vector to GPU
  for (int i = 0; i < BLOCKS; i++) {
    CHECK_CUDA(cudaMemcpyAsync(A_gpu + Nbegin[i], _A + Nbegin[i],
                               (Nend[i] - Nbegin[i]) * sizeof(float),
                               cudaMemcpyHostToDevice, upload_stream));
    CHECK_CUDA(cudaMemcpyAsync(B_gpu + Nbegin[i], _B + Nbegin[i],
                               (Nend[i] - Nbegin[i]) * sizeof(float),
                               cudaMemcpyHostToDevice, upload_stream));
    CHECK_CUDA(cudaEventRecord(upload_events[i], upload_stream));
  }

  // Launch kernel on a GPU
  for (int i = 0; i < BLOCKS; i++) {
    CHECK_CUDA(cudaStreamWaitEvent(calc_stream, upload_events[i]));
    dim3 gridDim((Nend[i] - Nbegin[i] + 512 - 1) / 512);
    dim3 blockDim(512);
    vecadd_kernel<<<gridDim, blockDim, 0, calc_stream>>>(
        Nend[i] - Nbegin[i], A_gpu + Nbegin[i], B_gpu + Nbegin[i],
        C_gpu + Nbegin[i]);
    CHECK_CUDA(cudaEventRecord(calc_events[i], calc_stream));
  }

  // Download C vector from GPU
  for (int i = 0; i < BLOCKS; i++) {
    CHECK_CUDA(cudaStreamWaitEvent(download_stream, calc_events[i]));
    CHECK_CUDA(cudaMemcpyAsync(_C + Nbegin[i], C_gpu + Nbegin[i],
                               (Nend[i] - Nbegin[i]) * sizeof(float),
                               cudaMemcpyDeviceToHost, download_stream));
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void vecadd_init(int N) {
  for (int i = 0; i < BLOCKS; i++) {
    Nbegin[i] = N / BLOCKS * i;
    Nend[i] = N / BLOCKS * (i + 1);
    if (i == BLOCKS - 1) Nend[i] = N;
  }

  // Allocate device memory
  CHECK_CUDA(cudaMalloc(&A_gpu, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B_gpu, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_gpu, N * sizeof(float)));

  // Create streams
  CHECK_CUDA(cudaStreamCreate(&upload_stream));
  CHECK_CUDA(cudaStreamCreate(&download_stream));
  CHECK_CUDA(cudaStreamCreate(&calc_stream));
  for (int i = 0; i < BLOCKS; i++) {
    CHECK_CUDA(cudaEventCreate(&upload_events[i]));
    CHECK_CUDA(cudaEventCreate(&calc_events[i]));
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void vecadd_cleanup(float *_A, float *_B, float *_C, int N) {
  // Free GPU memoryy
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));

  CHECK_CUDA(cudaStreamDestroy(upload_stream));
  CHECK_CUDA(cudaStreamDestroy(download_stream));
  CHECK_CUDA(cudaStreamDestroy(calc_stream));
  for (int i = 0; i < BLOCKS; i++) {
    CHECK_CUDA(cudaEventDestroy(upload_events[i]));
    CHECK_CUDA(cudaEventDestroy(calc_events[i]));
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
