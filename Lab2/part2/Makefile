CC = gcc
OPT3 = -O3
CFLAGS = -fopenmp -Wall -std=c99
LFLAGS = -lm

ALL = heat-omp
all: $(ALL)

misc.o: misc.c
	$(CC) -c $(CFLAGS) $(OPT3) $< -o $@

heat-omp: heat-omp.c solver-omp.c misc.o
	$(CC) $(CFLAGS) $(OPT3) $+ $(LFLAGS) -o $@

clean:
	rm -fr $(ALL) *.o

ultraclean:
	rm -fr $(ALL) *.o *.ppm sub*.o* sub*.e* *.prv *.pcf *.row time-heat-omp*boada*.txt