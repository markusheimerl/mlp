CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

train.out: mlp.o data.o train.o
	$(CC) mlp.o data.o train.o $(LDFLAGS) -o $@

mlp.o: mlp.c mlp.h
	$(CC) $(CFLAGS) -c mlp.c -o $@

data.o: data.c data.h
	$(CC) $(CFLAGS) -c data.c -o $@

train.o: train.c mlp.h data.h
	$(CC) $(CFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *.csv *.bin
	$(MAKE) -C gpu clean
	$(MAKE) -C bmlp clean