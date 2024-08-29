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
    half *buf = nullptr;
    bool is_gpu = false;

    Tensor(const vector<size_t> &shape_, bool gpu = true);
    ~Tensor();

    void to_gpu();
    void to_cpu();
    size_t num_elem();
};

typedef Tensor Parameter;
typedef Tensor Activation;