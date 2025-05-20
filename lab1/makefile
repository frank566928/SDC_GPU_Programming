CC = /usr/bin/clang++
CFLAGS = -fopenmp -O3 -ffast-math -march=native

all: my_program

my_program: lab1_opt.cpp
	$(CC) $(CFLAGS) -o my_program lab1_opt.cpp

run: my_program
	OMP_DISPLAY_ENV=VERBOSE ./my_program

clean:
	rm -f my_program

