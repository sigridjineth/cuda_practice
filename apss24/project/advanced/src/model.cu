#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

#include "layer.h"
#include "model.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

/* [Model Parameters]
 * _w: Weight parameter
 * _b: Bias parameter
 */
Parameter *mlp1_w, *mlp1_b;
Parameter *mlp2_w, *mlp2_b;
Parameter *convtrans1_w, *convtrans1_b;
Parameter *batchnorm1_w, *batchnorm1_b;
Parameter *convtrans2_w, *convtrans2_b;
Parameter *batchnorm2_w, *batchnorm2_b;
Parameter *convtrans3_w, *convtrans3_b;
Parameter *batchnorm3_w, *batchnorm3_b;
Parameter *convtrans4_w, *convtrans4_b;
Parameter *batchnorm4_w, *batchnorm4_b;
Parameter *convtrans5_w, *convtrans5_b;
Parameter *batchnorm5_w, *batchnorm5_b;
Parameter *convtrans6_w, *convtrans6_b;
Parameter *batchnorm6_w, *batchnorm6_b;
Parameter *conv_w, *conv_b;

void alloc_and_set_parameters(half_cpu *param, size_t param_size) {
    size_t pos = 0;

    mlp1_w = new Parameter({16384, 128}, param + pos);
    pos += 16384 * 128;
    mlp1_b = new Parameter({16384}, param + pos);
    pos += 16384;

    mlp2_w = new Parameter({4096, 16384}, param + pos);
    pos += 4096 * 16384;
    mlp2_b = new Parameter({4096}, param + pos);
    pos += 4096;

    convtrans1_w = new Parameter({1024, 512, 3, 3}, param + pos);
    pos += 1024 * 512 * 3 * 3;
    convtrans1_b = new Parameter({512}, param + pos);
    pos += 512;
    batchnorm1_w = new Parameter({512}, param + pos);
    pos += 512;
    batchnorm1_b = new Parameter({512}, param + pos);
    pos += 512;

    convtrans2_w = new Parameter({512, 256, 3, 3}, param + pos);
    pos += 512 * 256 * 3 * 3;
    convtrans2_b = new Parameter({256}, param + pos);
    pos += 256;
    batchnorm2_w = new Parameter({256}, param + pos);
    pos += 256;
    batchnorm2_b = new Parameter({256}, param + pos);
    pos += 256;

    convtrans3_w = new Parameter({256, 128, 3, 3}, param + pos);
    pos += 256 * 128 * 3 * 3;
    convtrans3_b = new Parameter({128}, param + pos);
    pos += 128;
    batchnorm3_w = new Parameter({128}, param + pos);
    pos += 128;
    batchnorm3_b = new Parameter({128}, param + pos);
    pos += 128;

    convtrans4_w = new Parameter({128, 64, 3, 3}, param + pos);
    pos += 128 * 64 * 3 * 3;
    convtrans4_b = new Parameter({64}, param + pos);
    pos += 64;
    batchnorm4_w = new Parameter({64}, param + pos);
    pos += 64;
    batchnorm4_b = new Parameter({64}, param + pos);
    pos += 64;

    convtrans5_w = new Parameter({64, 32, 3, 3}, param + pos);
    pos += 64 * 32 * 3 * 3;
    convtrans5_b = new Parameter({32}, param + pos);
    pos += 32;
    batchnorm5_w = new Parameter({32}, param + pos);
    pos += 32;
    batchnorm5_b = new Parameter({32}, param + pos);
    pos += 32;

    convtrans6_w = new Parameter({32, 32, 3, 3}, param + pos);
    pos += 32 * 32 * 3 * 3;
    convtrans6_b = new Parameter({32}, param + pos);
    pos += 32;
    batchnorm6_w = new Parameter({32}, param + pos);
    pos += 32;
    batchnorm6_b = new Parameter({32}, param + pos);
    pos += 32;

    conv_w = new Parameter({3, 32, 3, 3}, param + pos);
    pos += 3 * 32 * 3 * 3;
    conv_b = new Parameter({3}, param + pos);
    pos += 3;

    if (pos != param_size) {
        fprintf(stderr, "Parameter size mismatched: %zu vs %zu\n", pos, param_size);
        exit(1);
    }
}

void free_parameters() {
    delete mlp1_w;
    delete mlp1_b;
    delete mlp2_w;
    delete mlp2_b;
    delete convtrans1_w;
    delete convtrans1_b;
    delete batchnorm1_w;
    delete batchnorm1_b;
    delete convtrans2_w;
    delete convtrans2_b;
    delete batchnorm2_w;
    delete batchnorm2_b;
    delete convtrans3_w;
    delete convtrans3_b;
    delete batchnorm3_w;
    delete batchnorm3_b;
    delete convtrans4_w;
    delete convtrans4_b;
    delete batchnorm4_w;
    delete batchnorm4_b;
    delete convtrans5_w;
    delete convtrans5_b;
    delete batchnorm5_w;
    delete batchnorm5_b;
    delete convtrans6_w;
    delete convtrans6_b;
    delete batchnorm6_w;
    delete batchnorm6_b;
    delete conv_w;
    delete conv_b;
}

/* [Model Activations]
 * _a: Activation buffer
 */
Activation *linear1_a, *linear2_a;
Activation *reshape_a;
Activation *convtrans1_a, *batchnorm1_a;
Activation *convtrans2_a, *batchnorm2_a;
Activation *convtrans3_a, *batchnorm3_a;
Activation *convtrans4_a, *batchnorm4_a;
Activation *convtrans5_a, *batchnorm5_a;
Activation *convtrans6_a, *batchnorm6_a;
Activation *conv_a;

void alloc_activations() {
    linear1_a = new Activation({1, 16384});
    linear2_a = new Activation({1, 4096});
    reshape_a = new Activation({1, 1024, 2, 2});
    convtrans1_a = new Activation({1, 512, 4, 4});
    batchnorm1_a = new Activation({1, 512, 4, 4});
    convtrans2_a = new Activation({1, 256, 8, 8});
    batchnorm2_a = new Activation({1, 256, 8, 8});
    convtrans3_a = new Activation({1, 128, 16, 16});
    batchnorm3_a = new Activation({1, 128, 16, 16});
    convtrans4_a = new Activation({1, 64, 32, 32});
    batchnorm4_a = new Activation({1, 64, 32, 32});
    convtrans5_a = new Activation({1, 32, 64, 64});
    batchnorm5_a = new Activation({1, 32, 64, 64});
    convtrans6_a = new Activation({1, 32, 128, 128});
    batchnorm6_a = new Activation({1, 32, 128, 128});
    conv_a = new Activation({1, 3, 128, 128});
}

void free_activations() {
    delete linear1_a;
    delete linear2_a;
    delete reshape_a;
    delete convtrans1_a;
    delete batchnorm1_a;
    delete convtrans2_a;
    delete batchnorm2_a;
    delete convtrans3_a;
    delete batchnorm3_a;
    delete convtrans4_a;
    delete batchnorm4_a;
    delete convtrans5_a;
    delete batchnorm5_a;
    delete convtrans6_a;
    delete batchnorm6_a;
    delete conv_a;
}

void generate_images(half_cpu *input, half_cpu *output, size_t n_img) {
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // 입력 텐서 생성
    Tensor *z = new Tensor({n_img, LATENT_DIM}, input);
    z->to_device_async(stream);

    // 동적으로 Activation 버퍼 생성
    Activation *linear1_a_batch = new Activation({n_img, 16384});
    Activation *linear2_a_batch = new Activation({n_img, 4096});
    Activation *reshape_a_batch = new Activation({n_img, 1024, 2, 2});
    Activation *convtrans1_a_batch = new Activation({n_img, 512, 4, 4});
    Activation *batchnorm1_a_batch = new Activation({n_img, 512, 4, 4});
    Activation *convtrans2_a_batch = new Activation({n_img, 256, 8, 8});
    Activation *batchnorm2_a_batch = new Activation({n_img, 256, 8, 8});
    Activation *convtrans3_a_batch = new Activation({n_img, 128, 16, 16});
    Activation *batchnorm3_a_batch = new Activation({n_img, 128, 16, 16});
    Activation *convtrans4_a_batch = new Activation({n_img, 64, 32, 32});
    Activation *batchnorm4_a_batch = new Activation({n_img, 64, 32, 32});
    Activation *convtrans5_a_batch = new Activation({n_img, 32, 64, 64});
    Activation *batchnorm5_a_batch = new Activation({n_img, 32, 64, 64});
    Activation *convtrans6_a_batch = new Activation({n_img, 32, 128, 128});
    Activation *batchnorm6_a_batch = new Activation({n_img, 32, 128, 128});
    Activation *conv_a_batch = new Activation({n_img, 3, 128, 128});

    // 계산 시작 (기존 activation 버퍼 대신 새로 생성한 batch 버퍼 사용)
    Linear(z, mlp1_w, mlp1_b, linear1_a_batch, stream);
    Linear(linear1_a_batch, mlp2_w, mlp2_b, linear2_a_batch, stream);
    Reshape(linear2_a_batch, reshape_a_batch, stream);

    ConvTranspose2d(reshape_a_batch, convtrans1_w, convtrans1_b, convtrans1_a_batch, stream);
    BatchNorm2d(convtrans1_a_batch, batchnorm1_w, batchnorm1_b, batchnorm1_a_batch, stream);
    LeakyReLU(batchnorm1_a_batch, stream);

    ConvTranspose2d(batchnorm1_a_batch, convtrans2_w, convtrans2_b, convtrans2_a_batch, stream);
    BatchNorm2d(convtrans2_a_batch, batchnorm2_w, batchnorm2_b, batchnorm2_a_batch, stream);
    LeakyReLU(batchnorm2_a_batch, stream);

    ConvTranspose2d(batchnorm2_a_batch, convtrans3_w, convtrans3_b, convtrans3_a_batch, stream);
    BatchNorm2d(convtrans3_a_batch, batchnorm3_w, batchnorm3_b, batchnorm3_a_batch, stream);
    LeakyReLU(batchnorm3_a_batch, stream);

    ConvTranspose2d(batchnorm3_a_batch, convtrans4_w, convtrans4_b, convtrans4_a_batch, stream);
    BatchNorm2d(convtrans4_a_batch, batchnorm4_w, batchnorm4_b, batchnorm4_a_batch, stream);
    LeakyReLU(batchnorm4_a_batch, stream);

    ConvTranspose2d(batchnorm4_a_batch, convtrans5_w, convtrans5_b, convtrans5_a_batch, stream);
    BatchNorm2d(convtrans5_a_batch, batchnorm5_w, batchnorm5_b, batchnorm5_a_batch, stream);
    LeakyReLU(batchnorm5_a_batch, stream);

    ConvTranspose2d(batchnorm5_a_batch, convtrans6_w, convtrans6_b, convtrans6_a_batch, stream);
    BatchNorm2d(convtrans6_a_batch, batchnorm6_w, batchnorm6_b, batchnorm6_a_batch, stream);
    LeakyReLU(batchnorm6_a_batch, stream);

    Conv2d(batchnorm6_a_batch, conv_w, conv_b, conv_a_batch, stream);
    Tanh(conv_a_batch, stream);

    // 결과를 비동기적으로 호스트로 전송
    conv_a_batch->to_host_async(stream);

    // 스트림 동기화
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 결과를 출력 버퍼로 복사
    memcpy(output, conv_a_batch->buf, n_img * 3 * 128 * 128 * sizeof(half_cpu));

    // CUDA 스트림 정리
    CHECK_CUDA(cudaStreamDestroy(stream));

    // 임시 텐서 및 Activation 버퍼 메모리 해제
    delete z;
    delete linear1_a_batch;
    delete linear2_a_batch;
    delete reshape_a_batch;
    delete convtrans1_a_batch;
    delete batchnorm1_a_batch;
    delete convtrans2_a_batch;
    delete batchnorm2_a_batch;
    delete convtrans3_a_batch;
    delete batchnorm3_a_batch;
    delete convtrans4_a_batch;
    delete batchnorm4_a_batch;
    delete convtrans5_a_batch;
    delete batchnorm5_a_batch;
    delete convtrans6_a_batch;
    delete batchnorm6_a_batch;
    delete conv_a_batch;
}
