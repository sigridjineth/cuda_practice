#include <stdio.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>

#include "util.h"
#include "transpose.h"

static void print_help(const char* prog_name) {
  printf("Usage: %s [-pvh]  [-n num_iterations] M N\n", prog_name);
  printf("Options:\n");
  printf("  -p : print matrix data. (default: off)\n");
  printf("  -v : validate matrix transpose. (default: off)\n");
  printf("  -h : print this page.\n");
  printf("  -t : number of threads (default: 1)\n");
  printf("  -n : number of iterations (default: 1)\n");
  printf("   M : number of rows of matrix A. (default: 8)\n");
  printf("   N : number of columns of matrix A. (default: 8)\n");
}

static bool print_matrix = false;
static bool validation = false;
static int M = 8, N = 8;
static int num_iterations = 1;

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvht:n:")) != -1) {
    switch (c) {
      case 'p':
        print_matrix = true;
        break;
      case 'v':
        validation = true;
        break;
      case 'n':
        num_iterations = atoi(optarg);
        break;
      case 'h':
      default:
        print_help(argv[0]);
        exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0: M = atoi(argv[i]); break;
      case 1: N = atoi(argv[i]); break;
      default: break;
    }
  }
  printf("Options:\n");
  printf("  Problem size: M = %d, N = %d\n", M, N);
  printf("  Number of iterations: %d\n", num_iterations);
  printf("  Print matrix: %s\n", print_matrix ? "on" : "off");
  printf("  Validation: %s\n", validation ? "on" : "off");
  printf("\n");
}

int main(int argc, char **argv) {
  parse_opt(argc, argv);

  printf("Initializing... "); fflush(stdout);
  float *A = alloc_mat(M, N);
  float *B = alloc_mat(N, M);
  rand_mat(A, M, N);
  rand_mat(B, N, M);
  transpose_init(A, B, M, N);
  printf("done!\n");

  double elapsed_time_sum = 0;
  for (int i = 0; i < num_iterations; ++i) {
    printf("Calculating...(iter=%d) ", i); fflush(stdout);
    zero_mat(B, N, M);
    double start_time = get_time();
    transpose(A, B, M, N);
    double elapsed_time = get_time() - start_time;
    printf("%f sec\n", elapsed_time);
    elapsed_time_sum += elapsed_time;
  }

  if (print_matrix) {
    printf("MATRIX A:\n"); print_mat(A, M, N);
    printf("MATRIX B:\n"); print_mat(B, N, M);
  }

  transpose_cleanup(A, B, M, N);

  if (validation) {
    check_transpose(A, B, M, N);
  }

  double elapsed_time_avg = elapsed_time_sum / num_iterations;
  printf("Avg. time: %f sec\n", elapsed_time_avg);

  return 0;
}
