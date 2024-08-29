/* Last Updated: 24.08.29. 10:30 */
#include <cuda_runtime.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>

#include "model.h"

static size_t num_images = 2;
static size_t num_elems_per_image = (3 * 128 * 128);
static bool run_validation = false;

static char input_fname[100] = "./data/input_fp16.bin";
static char param_fname[100] = "/opt/apss24/project/param_fp16.bin";
static char output_fname[100] = "./data/output.bin";
static char answer_fname[100] = "/opt/apss24/project/answer_fp32.bin";

double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);

  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

void print_help() {
  fprintf(stdout,
          " Usage: ./main [-i 'pth'] [-p 'pth'] [-o 'pth'] [-a 'pth']"
          " [-n 'images'] [-v] [-h]\n");
  fprintf(stdout, " Options:\n");
  fprintf(stdout, 
          "  -i: Input binary path (default: ./data/input_fp16.bin)\n");
  fprintf(stdout,
          "  -p: Model parameter path (default: /opt/apss24/project/param_fp16.bin)\n");
  fprintf(stdout, 
          "  -o: Output binary path (default: ./data/output.bin)\n");
  fprintf(stdout, 
          "  -a: Answer binary path (default: /opt/apss24/project/answer_fp32.bin)\n");
  fprintf(stdout, "  -n: Number of input images (default: 1)\n");
  fprintf(stdout, "  -v: Enable validation (default: OFF)\n");
  fprintf(stdout, "  -h: Print manual and options (default: OFF)\n");
}

void parse_args(int argc, char **argv) {
  int args;
  while ((args = getopt(argc, argv, "i:o:a:p:n:vswh")) != -1) {
    switch (args) {
      case 'i': strcpy(input_fname, optarg); break;
      case 'o': strcpy(output_fname, optarg); break;
      case 'a': strcpy(answer_fname, optarg); break;
      case 'p': strcpy(param_fname, optarg); break;
      case 'n': num_images = atoi(optarg); break;
      case 'v': run_validation = true; break;
      case 'h':
        print_help();
        exit(0);
        break;
      default:
        print_help();
        exit(0);
        break;
    }
  }
  
  fprintf(stdout, "\n=============================================\n");
  fprintf(stdout, " Model: Variational AutoEncoder (VAE)\n");
  fprintf(stdout, "---------------------------------------------\n");
  fprintf(stdout, " Validation: %s\n", run_validation ? "ON" : "OFF");
  fprintf(stdout, " Number of images: %ld\n", num_images);
  fprintf(stdout, " Input binary path: %s\n", input_fname);
  fprintf(stdout, " Model parameter path: %s\n", param_fname);
  fprintf(stdout, " Answer binary path: %s\n", answer_fname);
  fprintf(stdout, " Output binary path: %s\n", output_fname);
  fprintf(stdout, "=============================================\n\n");
}

int validate(half_cpu *output, float *answer, int size_) {
  float threshold = 5e-1f;
  float max_min_err = 0.0f;

  for (int i = 0; i < size_; i++) {
    float abs_err = fabs(output[i] - answer[i]);
    float rel_err = (fabs(answer[i]) > 1e-8) ? abs_err / fabs(answer[i]) : abs_err;
    float min_err = fmin(abs_err, rel_err);
 
    max_min_err = fmax(max_min_err, min_err);

    if (max_min_err > threshold || std::isnan(output[i])) {
      return i;
    }
  }

  return -1; 
}

void *read_binary(const char *fname, size_t *size) {
  FILE *f = fopen(fname, "rb");
  if (f == NULL) {
    fprintf(stdout, "[ERROR] Cannot open file \'%s\'\n", fname);
    exit(-1);
  }

  fseek(f, 0, SEEK_END);
  size_t size_ = ftell(f);
  rewind(f);

  void *buf;
  cudaMallocHost(&buf, size_);
  size_t ret = fread(buf, 1, size_, f);
  if (ret == 0) {
    fprintf(stdout, "[ERROR] Cannot read file \'%s\'\n", fname);
    exit(-1);
  }
  fclose(f);

  if (size != NULL) *size = (size_t)(size_ / 2);  // 2 bytes per half

  return buf;
}

void write_binary(half_cpu *output, const char *filename, int size_) {
  FILE *f = (FILE *) fopen(filename, "wb");
  fwrite(output, sizeof(half_cpu), size_, f);
  fclose(f);
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  ////////////////////////////////////////////////////////////////////
  // INITIALIZATION                                                 //
  ////////////////////////////////////////////////////////////////////

  half_cpu *input = nullptr, *output = nullptr;
  half_cpu *param = nullptr;
  size_t param_size = 0;
	
  fprintf(stdout, "Initializing input and parameters...");
  fflush(stdout);

  /* Load input (size: [(num_images) * LATENT_DIM(=128)]) from file  */
  size_t input_size;
  input = (half_cpu *) read_binary(input_fname, &input_size);

  /* Allocate output (size: [(num_images) * (3 * 128 * 128)]) */
  cudaMallocHost((void**)&output, num_images * num_elems_per_image * sizeof(half_cpu));
  
  /* Initialize parameters and activations */
  param = (half_cpu *) read_binary(param_fname, &param_size);
  alloc_and_set_parameters(param, param_size);
  alloc_activations();

  fprintf(stdout, "Done!\n");

  ////////////////////////////////////////////////////////////////////
  // MODEL COMPUTATION                                              //
  ////////////////////////////////////////////////////////////////////

  double st = 0.0, et = 0.0;
  for (size_t i = 0; i < 4; i++) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }
  cudaSetDevice(0);

  fprintf(stdout, "Generating images...");
  fflush(stdout);
  
  st = get_time();

  /* Call the main computation (optimization target) of the program. */
	generate_images(input, output, num_images);
  
  for (size_t i = 0; i < 4; i++) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }
  cudaSetDevice(0);

  et = get_time();

  /* Print the result */
  fprintf(stdout, "Done!\n");
  fprintf(stdout, "Elapsed time: %lf (sec)\n", et - st);
  fprintf(stdout, "Throughput: %lf (images/sec)\n",
          num_images / (et - st));

  ////////////////////////////////////////////////////////////////////
  // FINALIZATION                                                   //
  ////////////////////////////////////////////////////////////////////

  /* Finalize parameters and activations */
  fprintf(stdout, "Finalizing...");
  free_parameters();
  free_activations();
  fprintf(stdout, "Done!\n");

  /* Save output */
  fprintf(stdout, "Saving output to %s...", output_fname);
  write_binary((half_cpu *) output, output_fname, num_images * num_elems_per_image);
  fprintf(stdout, "Done!\n");

  /* Validation */
  if (run_validation) {
    fprintf(stdout, "Validation...");
    float *answer = (float *) read_binary(answer_fname, NULL);
    int ret = validate(output, answer, num_images * num_elems_per_image);
    if (ret == -1) {
      fprintf(stdout, "PASSED!\n");
    } else {
      std::cout << "FAILED!\nFirst mismatch "
                << "at image[" << (int) (ret / num_elems_per_image) 
                << "], index[" << ret % num_elems_per_image << "] "
                << "(output[" << ret << "]=" << output[ret] 
                << " <-> answer[" << ret << "]=" << answer[ret] 
                << ")\n";
    }
  }

  return 0;
}
