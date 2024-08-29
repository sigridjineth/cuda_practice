#pragma once

#include <vector>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "half.hpp"

using std::vector;

typedef half_float::half half_cpu;
using namespace half_float::literal;

struct Tensor {
    size_t ndim = 0;
    vector<size_t> shape;
    half_cpu *buf = nullptr;
    half *d_buf = nullptr;  // GPU buffer

    Tensor(const vector<size_t> &shape_, size_t batch_size = 1);
    Tensor(const vector<size_t> &shape_, half_cpu *data);
    ~Tensor();

    size_t num_elem();
    void to_device(size_t batch_size);
    void to_host(size_t batch_size);
    void to_device_async(cudaStream_t stream, size_t batch_size);
    void to_host_async(cudaStream_t stream, size_t batch_size);
    void resize(const vector<size_t> &new_shape, size_t batch_size);
    void set_data(half_cpu *data, size_t batch_size);
};

class Parameter : public Tensor {
public:
    Parameter(const vector<size_t> &shape_, half_cpu *data);
};

typedef Tensor Activation;