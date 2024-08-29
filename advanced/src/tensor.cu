#include "tensor.h"
#include <cstring>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

Tensor::Tensor(const vector<size_t> &shape_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
    size_t N_ = num_elem();
    buf = (half_cpu *) calloc(N_, sizeof(half_cpu));
    CHECK_CUDA(cudaMalloc(&d_buf, N_ * sizeof(half)));
}

Tensor::Tensor(const vector<size_t> &shape_, half_cpu *buf_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
    size_t N_ = num_elem();
    buf = (half_cpu *) malloc(N_ * sizeof(half_cpu));
    memcpy(buf, buf_, N_ * sizeof(half_cpu));
    CHECK_CUDA(cudaMalloc(&d_buf, N_ * sizeof(half)));
    to_device();
}

Tensor::~Tensor() {
    if (buf != nullptr) free(buf);
    if (d_buf != nullptr) CHECK_CUDA(cudaFree(d_buf));
}

size_t Tensor::num_elem() {
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) { size *= shape[i]; }
    return size;
}

void Tensor::to_device() {
    size_t N_ = num_elem();
    CHECK_CUDA(cudaMemcpy(d_buf, buf, N_ * sizeof(half), cudaMemcpyHostToDevice));
}

void Tensor::to_host() {
    size_t N_ = num_elem();
    CHECK_CUDA(cudaMemcpy(buf, d_buf, N_ * sizeof(half), cudaMemcpyDeviceToHost));
}