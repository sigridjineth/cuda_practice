#include <cstdio>
#include <cstdlib>
#include <ctime>

double get_current_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

__global__ void vec_accum_kernel(const int *A, const int *B, int *C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) C[i] += A[i] + B[i];
}

int main() {
  int N = 16384;
  int num_iters = 20;
  int calls_per_iter = 1000;

  float *A = (float *) malloc(N * sizeof(float));
  float *B = (float *) malloc(N * sizeof(float));
  float *C = (float *) malloc(N * sizeof(float));

  for (int i = 0; i < N; i++) {
    A[i] = rand() / RAND_MAX;
    B[i] = rand() / RAND_MAX;
  }

  int *A_gpu, *B_gpu, *C_gpu;
  CHECK_CUDA(cudaMalloc(&A_gpu, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B_gpu, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_gpu, N * sizeof(float)));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  /* Run job for num_iters times. Each iteration consists of calls_per_iter
   * small kernel calls */
  {
    CHECK_CUDA(cudaMemcpy(A_gpu, A, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_gpu, B, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(C_gpu, 0, N * sizeof(float)));

    double start_time = get_current_time();
    for (int i = 0; i < num_iters; i++) {
      // repeat work for num_iters times
      for (int j = 0; j < calls_per_iter; j++) {
        // each iteration consists of calls_per_iter short kernel calls
        dim3 gridDim((N + 1024 - 1) / 1024);
        dim3 blockDim(1024);
        vec_accum_kernel<<<gridDim, blockDim, 0, stream>>>(A_gpu, B_gpu, C_gpu,
                                                           N);
      }
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    double end_time = get_current_time();

    CHECK_CUDA(cudaMemcpy(C, C_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Elapsed time without CUDA Graph: %.3f ms\n",
           (end_time - start_time) * 1e3);
  }

  /* Run the same job with CUDA Graph */
  {
    CHECK_CUDA(cudaMemcpy(A_gpu, A, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_gpu, B, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(C_gpu, 0, N * sizeof(float)));

    bool graph_created = false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    // Time for num_calls kernel calls
    double start_time = get_current_time();
    for (int i = 0; i < num_iters; i++) {
      if (!graph_created) {
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        for (int j = 0; j < calls_per_iter; j++) {
          dim3 gridDim((N + 1024 - 1) / 1024);
          dim3 blockDim(1024);
          vec_accum_kernel<<<gridDim, blockDim, 0, stream>>>(A_gpu, B_gpu,
                                                             C_gpu, N);
        }
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        graph_created = true;
      } 
      else {
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);
      }
    }
    double end_time = get_current_time();

    CHECK_CUDA(cudaMemcpy(C, C_gpu, N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Elapsed time with CUDA Graph: %.3f ms\n",
           (end_time - start_time) * 1e3);
  }

  return 0;
}
