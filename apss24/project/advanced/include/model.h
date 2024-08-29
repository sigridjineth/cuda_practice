#pragma once

#include "tensor.h"


/* Model configuration */
#define LATENT_DIM 128

void alloc_and_set_parameters(half_cpu *param, size_t param_size);
void alloc_activations();
void generate_images(half_cpu *input, half_cpu *output, size_t n_img);
void free_parameters();
void free_activations();