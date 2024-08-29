#include <cstdio>

#include "integral.h"

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
static double *pi_gpu;

static double f(double x) { return 4.0 / (1 + x * x); }

double integral_naive(size_t num_intervals) {
  double dx, sum;
  dx = (1.0 / (double) num_intervals);
  sum = 0.0f;
  for (size_t i = 0; i < num_intervals; i++) { sum += f(i * dx) * dx; }
  return sum;
}

double integral(size_t num_intervals) {
  double pi_value = 0.0;
  // Remove this line after you complete the matmul on GPU
  pi_value = integral_naive(num_intervals);

  // (TODO) Launch kernel on a GPU

  // (TODO) Download pi_value from GPU

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());

  return pi_value;
}

void integral_init(size_t num_intervals) {
  // (TODO) Allocate device memory

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void integral_cleanup() {
  // (TODO) Free device memory

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
