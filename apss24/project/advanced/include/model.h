#pragma once

#include "tensor.h"

/* Model configuration */
#define LATENT_DIM 128

// 파라미터 변수 선언
extern Parameter *mlp1_w, *mlp1_b;
extern Parameter *mlp2_w, *mlp2_b;
extern Parameter *convtrans1_w, *convtrans1_b;
extern Parameter *batchnorm1_w, *batchnorm1_b;
extern Parameter *convtrans2_w, *convtrans2_b;
extern Parameter *batchnorm2_w, *batchnorm2_b;
extern Parameter *convtrans3_w, *convtrans3_b;
extern Parameter *batchnorm3_w, *batchnorm3_b;
extern Parameter *convtrans4_w, *convtrans4_b;
extern Parameter *batchnorm4_w, *batchnorm4_b;
extern Parameter *convtrans5_w, *convtrans5_b;
extern Parameter *batchnorm5_w, *batchnorm5_b;
extern Parameter *convtrans6_w, *convtrans6_b;
extern Parameter *batchnorm6_w, *batchnorm6_b;
extern Parameter *conv_w, *conv_b;

void alloc_and_set_parameters(half_cpu *param, size_t param_size);
void alloc_activations();
void generate_images(half_cpu *input, half_cpu *output, size_t n_img);
void free_parameters();
void free_activations();