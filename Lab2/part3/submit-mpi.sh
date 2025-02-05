#!/bin/bash

#SBATCH --job-name=submit-mpi.sh
#SBATCH -D .
#SBATCH --output=submit-mpi.sh.o%j
#SBATCH --error=submit-mpi.sh.e%j
## #SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1


PROGRAM=heat-mpi
procs=4

input=test.dat


HOST=$(echo $HOSTNAME | cut -f 1 -d'.')

if [ ${HOST} = 'boada-6' ] || [ ${HOST} = 'boada-7' ] || [ ${HOST} == 'boada-8' ]
then
    echo "Use sbatch to execute this script"
    exit 0
fi

USAGE="\n USAGE: ./submit-omp.sh [numthreads] \n
        numthreads  -> OPTIONAL: Number of threads in parallel execution\n
                                    (defaults to using 8 threads)\n"

if ( test $# -gt 1)
#if (test $# -lt 1 || test $# -gt 1)
then
        echo -e $USAGE
        exit 0
fi

if (test $# -eq 1 )
then
        procs=$1
fi


#make clean
make $PROGRAM

mpirun.mpich -np $procs ./$PROGRAM $input
