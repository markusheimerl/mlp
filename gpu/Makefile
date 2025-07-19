CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto
CUDAFLAGS = --cuda-gpu-arch=sm_87 -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

# Object files
OBJS = mlp.o train.o

# Default target
train.out: $(OBJS)
	$(CC) $(CFLAGS) $(CUDAFLAGS) $(OBJS) $(CUDALIBS) $(LDFLAGS) -o $@

# Individual object files
mlp.o: mlp.c mlp.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c mlp.c -o $@

train.o: train.c mlp.h ../data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

# Keep old target for compatibility
mlp.out: train.out
	cp train.out mlp.out

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *.csv *.bin
