CC = gcc
MPICC = mpicc
CFLAGS = -O3 -Wall -std=c99

ALL = heat-mpi
all: $(ALL)

misc.o: misc.c
	$(CC) -c $(CFLAGS) $< -o $@

heat-mpi: heat-mpi.c solver-mpi.c misc.o
	$(MPICC) $(CFLAGS) -o $@ $+ -lm

clean:
	rm -fr $(ALL) *.o *.mpi *~ *.ppm *.ps *.txt sub*.e* sub*.o* *.out *.prv *.pcf *.row TRACE.mpits set-0 *.sym
