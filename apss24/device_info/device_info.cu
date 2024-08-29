#include <cstdio>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

int main() {
  int count;
  CHECK_CUDA(cudaGetDeviceCount(&count));

  printf("Number of devices: %d\n", count);
  cudaDeviceProp props[4];
  for (int i = 0; i < count; ++i) {
    printf("\tdevice %d:\n", i);
    // TODO: get and print device properties
  }

  return 0;
}
