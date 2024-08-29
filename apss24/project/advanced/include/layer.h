#pragma once

#include "tensor.h"
#include <cuda_runtime.h>

/* Elementwise operations */
void LeakyReLU(Tensor *inout, cudaStream_t stream, size_t batch_size);
void Tanh(Tensor *inout, cudaStream_t stream, size_t batch_size);

/* Matmul operations */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out, cudaStream_t stream, size_t batch_size);

/* Data movement operations */
void Reshape(Tensor *in, Tensor *out, cudaStream_t stream, size_t batch_size);

/* Convolutional operations */
void ConvTranspose2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out, cudaStream_t stream, size_t batch_size);
void Conv2d(Tensor *in, Tensor *w, Tensor *b, Tensor *out, cudaStream_t stream, size_t batch_size);

/* Other operations */
void BatchNorm2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out, cudaStream_t stream, size_t batch_size);

/* CUDA stream management */
void initializeCUDA();
void finalizeCUDA();