#include <cstdio>

__global__ void welcome_kernel() {
  printf("(Device) Welcome to Accelerator Programming School!\n");
}

void welcome() {
  printf("(Host) Welcome to Accelerator Programming School!\n");
  welcome_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}
