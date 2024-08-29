#include <cstdio>

#include "integral.h"

#define THREADS_PER_BLOCK 1024
#define ELEMENTS_PER_BLOCK (THREADS_PER_BLOCK * 2)

#define CHECK_CUDA(f)                                                      \
  {                                                                        \
    cudaError_t err = (f);                                                 \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__, \
              err, cudaGetErrorString(err));                               \
      exit(1);                                                             \
    }                                                                      \
  }

// Device(GPU) pointers
static double *output_cpu;
static double *output_gpu;

static __device__ double f(double x) { return 4.0 / (1 + x * x); }

__global__ void integral_kernel(double *output, size_t N) {
  extern __shared__ double L[];

  unsigned int tid = threadIdx.x;
  unsigned int offset = blockIdx.x * blockDim.x * 2;
  unsigned int stride = blockDim.x;

  double dx = 1.0 / (double) N;
  L[tid] = 0;

  unsigned int x1 = tid + offset;
  unsigned int x2 = tid + stride + offset;
  if (x1 < N) L[tid] += f(x1 * dx) * dx;
  if (x2 < N) L[tid] += f(x2 * dx) * dx;
  __syncthreads();

  for (stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) L[tid] += L[tid + stride];
    __syncthreads();
  }

  if (tid == 0) output[blockIdx.x] = L[0];
}

double integral(size_t num_intervals) {
  double pi_value = 0.0;

  size_t output_elements =
      (num_intervals + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
  dim3 gridDim(output_elements);
  dim3 blockDim(THREADS_PER_BLOCK);
  integral_kernel<<<gridDim, blockDim, THREADS_PER_BLOCK * sizeof(double), 0>>>(
      output_gpu, num_intervals);

  CHECK_CUDA(cudaMemcpy(output_cpu, output_gpu,
                        output_elements * sizeof(double),
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < output_elements; i++) { pi_value += output_cpu[i]; }
  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());

  return pi_value;
}

void integral_init(size_t num_intervals) {
  CHECK_CUDA(cudaMalloc(&output_gpu, (num_intervals + ELEMENTS_PER_BLOCK - 1) /
                                         ELEMENTS_PER_BLOCK * sizeof(double)));
  output_cpu = (double *) malloc((num_intervals + ELEMENTS_PER_BLOCK - 1) /
                                 ELEMENTS_PER_BLOCK * sizeof(double));

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void integral_cleanup() {
  CHECK_CUDA(cudaFree(output_gpu));

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
