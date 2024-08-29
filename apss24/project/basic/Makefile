CXX=g++
CUX=/usr/local/cuda/bin/nvcc

CFLAGS=-std=c++14 -O3 -Wall -march=native -mavx2 -mfma -mno-avx512f -fopenmp -I/usr/local/cuda/include -Iinclude
CUDA_CFLAGS:=$(foreach option, $(CFLAGS), -Xcompiler=$(option))
LDFLAGS=-pthread -L/usr/local/cuda/lib64
LDLIBS=-lstdc++ -lcudart -lm

TARGET=main
OBJECTS=obj/main.o obj/model.o obj/tensor.o obj/layer.o


all: $(TARGET)

$(TARGET): create_obj $(OBJECTS) 
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)


obj/%.o: src/%.cpp
	$(CXX) $(CFLAGS) -c -o $@ $^

obj/%.o: src/%.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)

run:
	sh ./run.sh

create_obj:
	mkdir -p obj