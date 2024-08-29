#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nvml.h>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

#define CHECK_NVML(call)                                                    \
  {                                                                        \
    auto status = static_cast<nvmlReturn_t>(call);                         \
    if (status != NVML_SUCCESS)                                            \
      fprintf(stderr, "NVML error at [%s:%d] %d %s\n", __FILE__, __LINE__, \
              status, nvmlErrorString(status));                            \
  }

int main() {
  int count;
  CHECK_CUDA(cudaGetDeviceCount(&count));
  CHECK_NVML(nvmlInit());

  printf("Number of devices: %d\n", count);
  cudaDeviceProp props[4];
  for (int i = 0; i < count; ++i) {
    printf("\tdevice %d:\n", i);
    CHECK_CUDA(cudaGetDeviceProperties(&props[i], i));
    printf("\t\tname: %s\n", props[i].name);

    nvmlDevice_t device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(i, &device));

    nvmlMemory_t memory;
    CHECK_NVML(nvmlDeviceGetMemoryInfo(device, &memory));
    printf("\t\tMemory:\n");
    printf("\t\t\ttotal: %lu B\n", memory.total);
    printf("\t\t\tfree: %lu B\n", memory.free);
    printf("\t\t\tused: %lu B\n", memory.used);

    unsigned int power;
    CHECK_NVML(nvmlDeviceGetPowerUsage(device, &power));
    printf("\t\tPower usage: %u W\n", power / 1000);

    unsigned int temp;
    CHECK_NVML(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp));
    printf("\t\tTemperature: %u C\n", temp);

    nvmlUtilization_t utilization;
    CHECK_NVML(nvmlDeviceGetUtilizationRates(device, &utilization));
    printf("\t\tGPU utilization: %u %%\n", utilization.gpu);
    printf("\t\tMemory utilization: %u %%\n", utilization.memory);
  }

  return 0;
}
