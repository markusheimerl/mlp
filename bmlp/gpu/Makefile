CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto

ARCH ?= sm_87
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

train.out: bmlp.o data.o train.o
	$(CC) bmlp.o data.o train.o $(CUDALIBS) $(LDFLAGS) -o $@

bmlp.o: bmlp.c bmlp.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c bmlp.c -o $@

data.o: ../../data.c ../../data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c ../../data.c -o $@

train.o: train.c bmlp.h ../../data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *.csv *.bin