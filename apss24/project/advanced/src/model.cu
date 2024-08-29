#include "model.h"
#include "layer.h"
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

void generate_images(half_cpu *input, half_cpu *output, size_t n_img) {
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // 입력 텐서 생성
    Tensor *z = new Tensor({n_img, LATENT_DIM}, input);
    z->to_device_async(stream);

    // 출력 및 중간 결과를 위한 텐서 생성
    Tensor *linear1_a = new Tensor({n_img, 16384});
    Tensor *linear2_a = new Tensor({n_img, 4096});
    Tensor *reshape_a = new Tensor({n_img, 1024, 2, 2});
    Tensor *convtrans1_a = new Tensor({n_img, 512, 4, 4});
    Tensor *batchnorm1_a = new Tensor({n_img, 512, 4, 4});
    Tensor *convtrans2_a = new Tensor({n_img, 256, 8, 8});
    Tensor *batchnorm2_a = new Tensor({n_img, 256, 8, 8});
    Tensor *convtrans3_a = new Tensor({n_img, 128, 16, 16});
    Tensor *batchnorm3_a = new Tensor({n_img, 128, 16, 16});
    Tensor *convtrans4_a = new Tensor({n_img, 64, 32, 32});
    Tensor *batchnorm4_a = new Tensor({n_img, 64, 32, 32});
    Tensor *convtrans5_a = new Tensor({n_img, 32, 64, 64});
    Tensor *batchnorm5_a = new Tensor({n_img, 32, 64, 64});
    Tensor *convtrans6_a = new Tensor({n_img, 32, 128, 128});
    Tensor *batchnorm6_a = new Tensor({n_img, 32, 128, 128});
    Tensor *conv_a = new Tensor({n_img, 3, 128, 128});

    // 계산 시작 (모든 연산을 GPU에서 수행)
    Linear(z, mlp1_w, mlp1_b, linear1_a, stream);
    Linear(linear1_a, mlp2_w, mlp2_b, linear2_a, stream);
    Reshape(linear2_a, reshape_a, stream);

    ConvTranspose2d(reshape_a, convtrans1_w, convtrans1_b, convtrans1_a, stream);
    BatchNorm2d(convtrans1_a, batchnorm1_w, batchnorm1_b, batchnorm1_a, stream);
    LeakyReLU(batchnorm1_a, stream);

    ConvTranspose2d(batchnorm1_a, convtrans2_w, convtrans2_b, convtrans2_a, stream);
    BatchNorm2d(convtrans2_a, batchnorm2_w, batchnorm2_b, batchnorm2_a, stream);
    LeakyReLU(batchnorm2_a, stream);

    ConvTranspose2d(batchnorm2_a, convtrans3_w, convtrans3_b, convtrans3_a, stream);
    BatchNorm2d(convtrans3_a, batchnorm3_w, batchnorm3_b, batchnorm3_a, stream);
    LeakyReLU(batchnorm3_a, stream);

    ConvTranspose2d(batchnorm3_a, convtrans4_w, convtrans4_b, convtrans4_a, stream);
    BatchNorm2d(convtrans4_a, batchnorm4_w, batchnorm4_b, batchnorm4_a, stream);
    LeakyReLU(batchnorm4_a, stream);

    ConvTranspose2d(batchnorm4_a, convtrans5_w, convtrans5_b, convtrans5_a, stream);
    BatchNorm2d(convtrans5_a, batchnorm5_w, batchnorm5_b, batchnorm5_a, stream);
    LeakyReLU(batchnorm5_a, stream);

    ConvTranspose2d(batchnorm5_a, convtrans6_w, convtrans6_b, convtrans6_a, stream);
    BatchNorm2d(convtrans6_a, batchnorm6_w, batchnorm6_b, batchnorm6_a, stream);
    LeakyReLU(batchnorm6_a, stream);

    Conv2d(batchnorm6_a, conv_w, conv_b, conv_a, stream);
    Tanh(conv_a, stream);

    // 결과를 비동기적으로 호스트로 전송
    conv_a->to_host_async(stream);

    // 스트림 동기화
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 결과를 출력 버퍼로 복사
    memcpy(output, conv_a->buf, n_img * 3 * 128 * 128 * sizeof(half_cpu));

    // 메모리 해제
    delete z;
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

    CHECK_CUDA(cudaStreamDestroy(stream));
}