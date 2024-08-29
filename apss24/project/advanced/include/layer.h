#pragma once

#include "tensor.h"
#include <cuda_runtime.h>

/* Elementwise operations */
void LeakyReLU(Tensor *inout, cudaStream_t stream);
void Tanh(Tensor *inout, cudaStream_t stream);

/* Matmul operations */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out, cudaStream_t stream);

/* Data movement operations */
void Reshape(Tensor *in, Tensor *out, cudaStream_t stream);

/* Convolutional operations */
void ConvTranspose2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out, cudaStream_t stream);
void Conv2d(Tensor *in, Tensor *w, Tensor *b, Tensor *out, cudaStream_t stream);

/* Other operations */
void BatchNorm2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out, cudaStream_t stream);

/* Fused operations */
void ConvTranspose2dBatchNormLeakyReLU(Tensor *in, Tensor *weight, Tensor *bias,
                                       Tensor *bn_weight, Tensor *bn_bias,
                                       Tensor *out, cudaStream_t stream);

/* CUDA stream management */
void initializeCUDA();
void finalizeCUDA();