#!/bin/bash

#SBATCH --job-name=submit-heat-seq.sh
#SBATCH -D .
#SBATCH --output=submit-heat-seq.sh.o%j
#SBATCH --error=submit-heat-seq.sh.e%j
## #SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


PROGRAM=heat

make $PROGRAM

./$PROGRAM test.dat
