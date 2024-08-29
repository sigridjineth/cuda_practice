#pragma once

double get_time();

void check_transpose(float *A, float *B, int M, int N);

void print_mat(float *m, int R, int C);

float* alloc_mat(int R, int C);

void rand_mat(float *m, int R, int C);

void zero_mat(float *m, int R, int C);
