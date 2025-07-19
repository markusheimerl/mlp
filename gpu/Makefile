CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto
CUDAFLAGS = --cuda-gpu-arch=sm_87 -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

train.out: mlp.o data.o train.o
	$(CC) mlp.o data.o train.o $(CUDALIBS) $(LDFLAGS) -o $@

mlp.o: mlp.c mlp.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c mlp.c -o $@

data.o: ../data.c ../data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c ../data.c -o $@

train.o: train.c mlp.h ../data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *.csv *.bin