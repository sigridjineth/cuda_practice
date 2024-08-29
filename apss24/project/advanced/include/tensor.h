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
  half_cpu *buf = nullptr; // float -> half

  Tensor(const vector<size_t> &shape_);
  Tensor(const vector<size_t> &shape_, half_cpu *buf_);
  ~Tensor();

  size_t num_elem();
};

typedef Tensor Parameter;
typedef Tensor Activation;