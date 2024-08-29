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
    size_t shape[4];
    half_cpu *buf = nullptr;
    half *d_buf = nullptr;  // GPU buffer

    Tensor(const vector<size_t> &shape_);
    Tensor(const vector<size_t> &shape_, half_cpu *buf_);
    ~Tensor();

    size_t num_elem();
    void to_device();
    void to_host();
    void to_device_async(cudaStream_t stream);
    void to_host_async(cudaStream_t stream);
};

typedef Tensor Parameter;
typedef Tensor Activation;