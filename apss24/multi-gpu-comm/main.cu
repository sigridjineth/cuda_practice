#include <cstdio>
#include <nccl.h>
#include <sys/time.h>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

#define CHECK_NCCL(call)                                               \
  do {                                                                 \
    ncclResult_t status_ = call;                                       \
    if (status_ != ncclSuccess && status_ != ncclInProgress) {         \
      fprintf(stderr, "NCCL error (%s:%d): %s\n", __FILE__, __LINE__,  \
              ncclGetErrorString(status_));                            \
      exit(EXIT_FAILURE);                                                \
    }                                                                  \
  } while (0)

const int NUM_GPU = 4;
const int NITER = 3;
const size_t nbytes = 256 * 1024 * 1024; // 256MiB
int* sendbuf[NUM_GPU];
int* recvbuf[NUM_GPU];
cudaStream_t streams[NUM_GPU];

double SyncAllGPUsAndGetTime() {
  for (int i = 0; i < NUM_GPU; ++i) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

__global__ void FillBuffer(int* buf, size_t nbytes, size_t offset) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < nbytes / sizeof(int); i += stride) {
    buf[i] = i * sizeof(int) + offset;
  }
}

void InitBuffers() {
  for (int i = 0; i < NUM_GPU; ++i) {
    CHECK_CUDA(cudaSetDevice(i));
    FillBuffer<<<1, 1024>>>(sendbuf[i], nbytes, i * nbytes);
    CHECK_CUDA(cudaMemset(recvbuf[i], 0, nbytes * NUM_GPU));
  }
}

void CheckBuffers() {
  int* buf = (int*)malloc(nbytes * NUM_GPU);
  for (int i = 0; i < NUM_GPU; ++i) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpy(buf, recvbuf[i], nbytes * NUM_GPU, cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < nbytes * NUM_GPU / sizeof(int); ++j) {
      if (buf[j] != j * sizeof(int)) {
        printf("Incorrect! buf[%zu] should be %zu, but %d found\n", j, j * sizeof(int), buf[j]);
        goto end;
      }
    }
  }
  printf("Correct!\n");
  end: free(buf);
}

void AllGatherWithNCCL() {
  printf("[AllGather with NCCL]\n");

  ncclComm_t comms[NUM_GPU];
  int devlist[NUM_GPU];
  for (int i = 0; i < NUM_GPU; ++i) {
    devlist[i] = i;
  }
  CHECK_NCCL(ncclCommInitAll(comms, NUM_GPU, devlist));

  for (int iter = 0; iter < NITER; ++iter) {
    double st = SyncAllGPUsAndGetTime();
    for (int i = 0; i < NUM_GPU; ++i) {
      /*
       * TODO
       * Implement AllGather with NCCL here.
       */
    }
    double et = SyncAllGPUsAndGetTime();
    double gbps = nbytes * (NUM_GPU - 1) / (et - st) / 1e9;
    printf("[Iter %d] %f sec (Effective bandwidth %f GB/s)\n", iter, et - st, gbps);
  }

  for (int i = 0; i < NUM_GPU; ++i) {
    CHECK_NCCL(ncclCommDestroy(comms[i]));
  }
}

void AllGatherWithMemcpy() {
  printf("[AllGather with Memcpy]\n");

  int canAccessPeer;
  for (int i = 0; i < NUM_GPU; ++i) {
    for (int j = 0; j < NUM_GPU; ++j) {
      cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
      if (canAccessPeer == 1) {
        cudaSetDevice(i);
        cudaDeviceEnablePeerAccess(j, 0);
      }
    }
  }

  for (int iter = 0; iter < NITER; ++iter) {
    double st = SyncAllGPUsAndGetTime();
    for (int i = 0; i < NUM_GPU; ++i) {
      /*
       * TODO
       * Implement AllGather using memcpy here.
       */
    }
    double et = SyncAllGPUsAndGetTime();
    double gbps = nbytes * (NUM_GPU - 1) / (et - st) / 1e9;
    printf("[Iter %d] %f sec (Effective bandwidth %f GB/s)\n", iter, et - st, gbps);
  }
}

int main(int argc, char **argv) {
  for (int i = 0; i < NUM_GPU; ++i) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMalloc(&sendbuf[i], nbytes));
    CHECK_CUDA(cudaMalloc(&recvbuf[i], nbytes * NUM_GPU));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
  }

  InitBuffers();
  AllGatherWithNCCL();
  CheckBuffers();

  InitBuffers();
  AllGatherWithMemcpy();
  CheckBuffers();

  return 0;
}
