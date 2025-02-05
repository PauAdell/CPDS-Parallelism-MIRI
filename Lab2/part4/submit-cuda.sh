#!/bin/bash
# @ job_name            = heat-CUDA
# @ partition           = debug
# @ initialdir          = .
# @ output              = heat-CUDA.%j.out
# @ error               = heat-CUDA.%j.err
# @ total_tasks         = 1
# @ gpus_per_node       = 1
# @ wall_clock_limit    = 00:02:00

### Directives for SLURM
#SBATCH --job-name=heat-CUDA
#SBATCH -D .
#SBATCH --output=submit-heat-CUDA.sh.o%j
#SBATCH --error=submit-heat-CUDA.sh.e%j
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --gres=gpu:4

export PATH=/Soft/cuda/11.2.1/bin:$PATH

KERNEL=heat-CUDA

# Define threads per block
txb=8

echo "Running ${KERNEL}"

./${KERNEL} test.dat -t $txb
