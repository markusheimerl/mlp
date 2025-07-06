CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto
CUDAFLAGS = --cuda-gpu-arch=sm_87 -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

mlp.out: mlp.c
	$(CC) $(CFLAGS) $(CUDAFLAGS) $^ $(CUDALIBS) $(LDFLAGS) -o $@

run: mlp.out
	@time ./mlp.out

clean:
	rm -f *.out *.csv *.bin
