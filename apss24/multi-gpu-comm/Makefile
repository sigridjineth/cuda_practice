TARGET=main

NVCC=/usr/local/cuda/bin/nvcc

all: $(TARGET)

main: main.cu
	$(NVCC) -o $@ $^ -lnccl

clean:
	rm -rf $(TARGET)
