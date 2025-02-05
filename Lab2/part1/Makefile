CC = gcc
OPT3 = -O3
OPTG0 = -g -O0
CFLAGS = -Wall -std=c99
LFLAGS = -lm

ALL = heat
all: $(ALL)

misc.o: misc.c
	$(CC) -c $(OPT3) $(CFLAGS) $< -o $@

heat: heat.c solver.c misc.o
	$(CC) $(OPT3) $(CFLAGS) $+ $(LFLAGS) -o $@

clean:
	rm -fr $(ALL) *.o *.ppm sub*.e* sub*.o* *.xdot *.prv *.pcf *.row