CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

train.out: bmlp.o data.o train.o
	$(CC) bmlp.o data.o train.o $(LDFLAGS) -o $@

bmlp.o: bmlp.c bmlp.h
	$(CC) $(CFLAGS) -c bmlp.c -o $@

data.o: ../data.c ../data.h
	$(CC) $(CFLAGS) -c ../data.c -o $@

train.o: train.c bmlp.h ../data.h
	$(CC) $(CFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *.csv *.bin
	$(MAKE) -C gpu clean