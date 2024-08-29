#pragma once

#include <vector>
#include <cstdio>

#include "half.hpp" /* for half on CPU ('half_cpu') */
#include "cuda_fp16.h" /* for half on GPU ('half') */

using std::vector;

/* Namespace for half on CPU ('half_cpu') */
typedef half_float::half half_cpu;
using namespace half_float::literal; 


/* tensor.h 분석:

Tensor 구조체가 정의되어 있습니다.
차원 수(ndim)와 shape(최대 4차원)을 저장합니다.
데이터는 half_cpu (CPU용 half 정밀도 부동소수점) 타입의 포인터로 저장됩니다.
Tensor 생성자와 소멸자가 정의되어 있습니다.
num_elem() 함수로 총 원소 수를 계산할 수 있습니다.

 */
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