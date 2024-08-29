#pragma once

/*
 * layer.h 분석:

다양한 딥러닝 연산들이 함수로 정의되어 있습니다.
주요 연산들:

요소별 연산: LeakyReLU, Tanh
행렬 곱: Linear
데이터 변형: Reshape
컨볼루션 연산: ConvTranspose2d, Conv2d
정규화: BatchNorm2d


GPU 커널 예제로 LeakyReLU_cuda가 포함되어 있습니다.
 */
#include "tensor.h"

/* Elementwise operations */
void LeakyReLU(Tensor *inout);
void Tanh(Tensor *inout);

/* Matmul operations */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out);

/* Data movement operations */
void Reshape(Tensor *in, Tensor *out);

/* Convolutional operations */
void ConvTranspose2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out);
void Conv2d(Tensor *in, Tensor *w, Tensor *b, Tensor *out);

/* Other operations */
void BatchNorm2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out);

/* Example GPU kernel */
void LeakyReLU_cuda(Tensor *inout);

void ConvTranspose2dBatchNormLeakyReLU(Tensor *in, Tensor *weight, Tensor *bias,
                                       Tensor *bn_weight, Tensor *bn_bias, Tensor *out);