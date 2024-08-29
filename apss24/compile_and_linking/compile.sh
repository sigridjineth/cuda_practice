#!/bin/bash

# CUDA installation directory
CUDA_ROOT=/usr/local/cuda

# CUDA Compiler (nvcc)
CUX=${CUDA_ROOT}/bin/nvcc

# Compile file1.cpp
g++ -c -I${CUDA_ROOT}/include -o file1.o file1.cpp

# Compile file2.cu
${CUX} -c -o file2.o file2.cu

# Link file1.o and file2.o
g++ -o main file1.o file2.o -L${CUDA_ROOT}/lib64 -lcudart

