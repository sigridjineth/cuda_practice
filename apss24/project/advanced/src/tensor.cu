#include "tensor.h"
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

Tensor::Tensor(const vector<size_t> &shape_, bool gpu) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
    size_t N_ = num_elem();
    is_gpu = gpu;
    if (is_gpu) {
        CHECK_CUDA(cudaMalloc(&buf, N_ * sizeof(half)));
    } else {
        buf = (half *) calloc(N_, sizeof(half));
    }
}

Tensor::Tensor(const vector<size_t> &shape_, half_cpu *buf_, bool gpu) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
    size_t N_ = num_elem();
    is_gpu = gpu;
    if (is_gpu) {
        CHECK_CUDA(cudaMalloc(&buf, N_ * sizeof(half)));
        CHECK_CUDA(cudaMemcpy(buf, buf_, N_ * sizeof(half), cudaMemcpyHostToDevice));
    } else {
        buf = (half *) malloc(N_ * sizeof(half));
        memcpy(buf, buf_, N_ * sizeof(half));
    }
}

Tensor::~Tensor() {
    if (is_gpu) {
        if (buf != nullptr) CHECK_CUDA(cudaFree(buf));
    } else {
        if (buf != nullptr) free(buf);
    }
}

void Tensor::to_gpu() {
    if (!is_gpu) {
        size_t N_ = num_elem();
        half *gpu_buf;
        CHECK_CUDA(cudaMalloc(&gpu_buf, N_ * sizeof(half)));
        CHECK_CUDA(cudaMemcpy(gpu_buf, buf, N_ * sizeof(half), cudaMemcpyHostToDevice));
        free(buf);
        buf = gpu_buf;
        is_gpu = true;
    }
}

void Tensor::to_cpu() {
    if (is_gpu) {
        size_t N_ = num_elem();
        half *cpu_buf = (half *) malloc(N_ * sizeof(half));
        CHECK_CUDA(cudaMemcpy(cpu_buf, buf, N_ * sizeof(half), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(buf));
        buf = cpu_buf;
        is_gpu = false;
    }
}

size_t Tensor::num_elem() {
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) { size *= shape[i]; }
    return size;
}