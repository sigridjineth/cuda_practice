#include "util.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdbool>
#include <cmath>
#include <sys/time.h>
#include <omp.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void check_transpose(float *A, float *B, int M, int N) {
  printf("Validating...\n");

  float *B_ans = alloc_mat(N, M);
  zero_mat(B_ans, N, M);

#pragma omp parallel for num_threads(20)
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      B_ans[i * M + j] = A[j * N + i];
    }
  }

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      float b = B[i * M + j];
      float b_ans = B_ans[i * M + j];
      if (fabsf(b - b_ans) > eps && (b_ans == 0 || fabsf((b - b_ans) / b_ans) > eps)) {
        ++cnt;
        if (cnt <= thr)
          printf("B[%d][%d] : correct_value = %f, your_value = %f\n", i, j, b_ans, b);
        if (cnt == thr + 1)
          printf("Too many error, only first %d values are printed.\n", thr);
        is_valid = false;
      }
    }
  }

  if (is_valid) {
    printf("Result: VALID\n");
  } else {
    printf("Result: INVALID\n");
  }
}

void print_mat(float *m, int R, int C) {
  for (int i = 0; i < R; ++i) { 
    for (int j = 0; j < C; ++j) {
      printf("%+.3f ", m[i * C + j]);
    }
    printf("\n");
  }
}

float* alloc_mat(int R, int C) {
  float *m = (float *) aligned_alloc(32, sizeof(float) * R * C);
  return m;
}

void rand_mat(float *m, int R, int C) {
  for (int i = 0; i < R; i++) { 
    for (int j = 0; j < C; j++) {
      m[i * C + j] = (float) rand() / RAND_MAX - 0.5;
    }
  }
}

void zero_mat(float *m, int R, int C) {
  memset(m, 0, sizeof(float) * R * C);
}
