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

Tensor::Tensor(const vector<size_t> &shape_, size_t batch_size) {
    shape = shape_;
    ndim = shape.size();
    if (ndim == 0) {
        CHECK_CUDA(cudaMallocHost(&buf, batch_size * sizeof(half_cpu)));
        CHECK_CUDA(cudaMalloc(&d_buf, batch_size * sizeof(half)));
    } else {
        size_t N = num_elem() * batch_size;
        CHECK_CUDA(cudaMallocHost(&buf, N * sizeof(half_cpu)));
        CHECK_CUDA(cudaMalloc(&d_buf, N * sizeof(half)));
    }
}

Tensor::Tensor(const vector<size_t> &shape_, half_cpu *data) {
    shape = shape_;
    ndim = shape.size();
    size_t N = num_elem();
    CHECK_CUDA(cudaMallocHost(&buf, N * sizeof(half_cpu)));
    memcpy(buf, data, N * sizeof(half_cpu));
    CHECK_CUDA(cudaMalloc(&d_buf, N * sizeof(half)));
    to_device(1); // batch_size를 1로 설정
}

Tensor::~Tensor() {
    if (buf != nullptr) CHECK_CUDA(cudaFreeHost(buf));
    if (d_buf != nullptr) CHECK_CUDA(cudaFree(d_buf));
}

size_t Tensor::num_elem() {
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) { size *= shape[i]; }
    return size;
}

void Tensor::to_device(size_t batch_size) {
    size_t N = num_elem() * batch_size;
    CHECK_CUDA(cudaMemcpy(d_buf, buf, N * sizeof(half), cudaMemcpyHostToDevice));
}

void Tensor::to_host(size_t batch_size) {
    size_t N = num_elem() * batch_size;
    CHECK_CUDA(cudaMemcpy(buf, d_buf, N * sizeof(half), cudaMemcpyDeviceToHost));
}

void Tensor::to_device_async(cudaStream_t stream, size_t batch_size) {
    size_t N = num_elem() * batch_size;
    CHECK_CUDA(cudaMemcpyAsync(d_buf, buf, N * sizeof(half), cudaMemcpyHostToDevice, stream));
}

void Tensor::to_host_async(cudaStream_t stream, size_t batch_size) {
    size_t N = num_elem() * batch_size;
    CHECK_CUDA(cudaMemcpyAsync(buf, d_buf, N * sizeof(half), cudaMemcpyDeviceToHost, stream));
}

void Tensor::resize(const vector<size_t> &new_shape, size_t batch_size) {
    shape = new_shape;
    ndim = new_shape.size();
    size_t N = num_elem() * batch_size;
    CHECK_CUDA(cudaFreeHost(buf));
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaMallocHost(&buf, N * sizeof(half_cpu)));
    CHECK_CUDA(cudaMalloc(&d_buf, N * sizeof(half)));
}

void Tensor::set_data(half_cpu *data, size_t batch_size) {
    size_t N = num_elem() * batch_size;
    memcpy(buf, data, N * sizeof(half_cpu));
}

Parameter::Parameter(const vector<size_t> &shape_, half_cpu *data) : Tensor(shape_, 1) {
    size_t N = num_elem();
    memcpy(buf, data, N * sizeof(half_cpu));
    to_device(1);
}