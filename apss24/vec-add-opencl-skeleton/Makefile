TARGET=vec_add_opencl
OBJECTS=main.o vec_add_opencl.o

CPPFLAGS=-std=c++14 -O3 -Wall -march=native -I/usr/local/cuda/include

LDFLAGS=-L/usr/local/cuda/lib64
LDLIBS=-lstdc++ -lm -lOpenCL

CXX=g++

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CPPFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)
