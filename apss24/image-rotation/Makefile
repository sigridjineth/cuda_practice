TARGET=main
OBJECTS=main.o util.o image_rotation.o

CPPFLAGS=-std=c++14 -O3 -Wall -march=native -mavx2 -mfma -fopenmp -mno-avx512f
CPPFLAGS+=-I/usr/local/cuda/include
CUDA_CFLAGS:=$(foreach option, $(CPPFLAGS),-Xcompiler=$(option)) 
LDFLAGS=-lstdc++ -lcudart -lm
LDFLAGS+=-L/usr/local/cuda/lib64

CXX=g++
NVCC=/usr/local/cuda/bin/nvcc

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CPPFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c -o $@ $^

%.o: %.cu
	$(NVCC) $(CUDA_CFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)
