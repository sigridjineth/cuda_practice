#include <cmath>
#include <cstdio>
#include <ctime>

#include "vec_add.h"

static void alloc_vec(float **m, int N);
static void rand_vec(float *m, int N);
static void check_vec_add(float *A, float *B, float *C, int N);

int main() {
  srand(time(NULL));

  int N = 1024;
  float *A, *B, *C;
  alloc_vec(&A, N);
  alloc_vec(&B, N);
  alloc_vec(&C, N);
  rand_vec(A, N);
  rand_vec(B, N);

  vec_add_opencl(A, B, C, N);

  check_vec_add(A, B, C, N);

  return 0;
}

void alloc_vec(float **m, int N) {
  *m = (float *) malloc(sizeof(float) * N);
  if (*m == NULL) {
    printf("Failed to allocate memory for vector.\n");
    exit(0);
  }
}

void rand_vec(float *m, int N) {
  for (int i = 0; i < N; i++) { m[i] = (float) rand() / RAND_MAX - 0.5; }
}

void check_vec_add(float *A, float *B, float *C, int N) {
  printf("Validating...\n");

  float *C_ans;
  alloc_vec(&C_ans, N);
  for (int i = 0; i < N; ++i) { C_ans[i] = A[i] + B[i]; }

  int is_valid = 1;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int i = 0; i < N; ++i) {
    float c = C[i];
    float c_ans = C_ans[i];
    if (fabsf(c - c_ans) > eps &&
        (c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps)) {
      ++cnt;
      if (cnt <= thr)
        printf("C[%d] : correct_value = %f, your_value = %f\n", i, c_ans, c);
      if (cnt == thr + 1)
        printf("Too many error, only first %d values are printed.\n", thr);
      is_valid = 0;
    }
  }

  if (is_valid) {
    printf("Result: VALID\n");
  } else {
    printf("Result: INVALID\n");
  }
}