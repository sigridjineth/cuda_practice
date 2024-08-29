#pragma once

#include <vector>
#include <cstdio>

#include "half.hpp" /* for half on CPU ('half_cpu') */
#include "cuda_fp16.h" /* for half on GPU ('half') */

using std::vector;

/* Namespace for half on CPU ('half_cpu') */
typedef half_float::half half_cpu;
using namespace half_float::literal;

/* [Tensor Structure] */
struct Tensor {
    size_t ndim = 0;
    size_t shape[4];
    half_cpu *buf = nullptr;
    half *d_buf = nullptr;  // GPU buffer
    bool is_gpu = false;

    Tensor(const vector<size_t> &shape_, bool gpu = false);
    Tensor(const vector<size_t> &shape_, half_cpu *buf_, bool gpu = false);
    ~Tensor();

    void to_device();
    void to_host();
    size_t num_elem();
};

typedef Tensor Parameter;
typedef Tensor Activation;