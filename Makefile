CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -static -lopenblas -lm -flto

# Object files
OBJS = mlp.o data.o train.o

# Default target
train.out: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LDFLAGS) -o $@

# Individual object files
mlp.o: mlp.c mlp.h
	$(CC) $(CFLAGS) -c mlp.c -o $@

data.o: data.c data.h
	$(CC) $(CFLAGS) -c data.c -o $@

train.o: train.c mlp.h data.h
	$(CC) $(CFLAGS) -c train.c -o $@

# Keep old target for compatibility
mlp.out: train.out
	cp train.out mlp.out

run: train.out
	@time ./train.out
	
clean:
	rm -f *.out *.o *.csv *.bin
