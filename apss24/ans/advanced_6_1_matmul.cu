#include <cstdio>
#include <nvToolsExt.h>
#include <cuda_profiler_api.h>

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

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                              int K) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if (j >= N || i >= M) return;

  // init C
  C[i * N + j] = 0;

  float a0, a1, a2, a3, a4, a5, a6, a7;
  float b0, b1, b2, b3, b4, b5, b6, b7;
  int k;

  // loop unrolling
  for (k = 0; k + 7 < K; k += 8) {
    a0 = A[i * K + (k + 0)];
    a1 = A[i * K + (k + 1)];
    a2 = A[i * K + (k + 2)];
    a3 = A[i * K + (k + 3)];
    a4 = A[i * K + (k + 4)];
    a5 = A[i * K + (k + 5)];
    a6 = A[i * K + (k + 6)];
    a7 = A[i * K + (k + 7)];
    b0 = B[(k + 0) * N + j];
    b1 = B[(k + 1) * N + j];
    b2 = B[(k + 2) * N + j];
    b3 = B[(k + 3) * N + j];
    b4 = B[(k + 4) * N + j];
    b5 = B[(k + 5) * N + j];
    b6 = B[(k + 6) * N + j];
    b7 = B[(k + 7) * N + j];
    C[i * N + j] += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 +
                    a6 * b6 + a7 * b7;
  }

  // Deal with trailing k
  for (; k < K; k++) {
    C[i * N + j] += A[i * K + k] * B[k * N + j];
  }
}

// Color preset 
unsigned int nvtx_colors[16] = {
  0xFF00FF00, 0xFF008000, 0xFF00FFFF, 0xFF008080, 0xFF0000FF, 0xFF000080,
  0xFFFF00FF, 0xFF800080, 0xFFFFFFFF, 0xFFC0C0C0, 0xFF808080, 0xFF000000,
  0xFFFF0000, 0xFF800000, 0xFFFFFF00, 0xFF808000,
};

void nvtx_add_marker(const char *message) {
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message;
  eventAttrib.color = nvtx_colors[rand() % 16];
  nvtxMarkEx(&eventAttrib);
}

void nvtx_range_push(const char *message) {
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message;
  eventAttrib.color = nvtx_colors[rand() % 16];

  nvtxRangePushEx(&eventAttrib);
}

void nvtx_range_pop() {
  nvtxRangePop();
}

nvtxRangeId_t nvtx_range_start(const char *message) {
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = message;
  eventAttrib.color = nvtx_colors[rand() % 16];

  return nvtxRangeStartEx(&eventAttrib);
}

void nvtx_range_stop(nvtxRangeId_t nvtx_id) {
  nvtxRangeEnd(nvtx_id);
}

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  nvtxRangeId_t matmul_id = nvtx_range_start("matmul");
  nvtxRangeId_t id = nvtx_range_start("Upload data");
  CHECK_CUDA(
      cudaMemcpy(A_gpu, _A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(B_gpu, _B, sizeof(float) * K * N, cudaMemcpyHostToDevice));
  nvtx_range_stop(id);

  id = nvtx_range_start("Run kernel");
  dim3 griddim((N + 16 - 1) / 16, (M + 16 - 1) / 16);
  dim3 blockdim(16, 16);
  matmul_kernel<<<griddim, blockdim>>>(A_gpu, B_gpu, C_gpu, M, N, K);
  CHECK_CUDA(cudaDeviceSynchronize());
  nvtx_range_stop(id);

  id = nvtx_range_start("Download data");
  CHECK_CUDA(
      cudaMemcpy(_C, C_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
  nvtx_range_stop(id);

  // Dummy ranges
  nvtx_range_push("Overlapping Range0");
  nvtx_range_push("Overlapping Range1");
  nvtx_range_push("Overlapping Range2");
  nvtx_range_pop();
  nvtx_range_pop();
  nvtx_range_pop();

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());

  // Dummy markers
  nvtxMarkA("End of matmul");
  nvtx_add_marker("End of matmul again");
  nvtx_range_stop(matmul_id);
}

void matmul_init(int M, int N, int K) {
  CHECK_CUDA(cudaMalloc((void **) &A_gpu, sizeof(float) * M * K));
  CHECK_CUDA(cudaMalloc((void **) &B_gpu, sizeof(float) * K * N));
  CHECK_CUDA(cudaMalloc((void **) &C_gpu, sizeof(float) * M * N));

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
