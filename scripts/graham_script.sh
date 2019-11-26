#!/bin/bash
#SBATCH --account=rrg-rgmelko-ab
#SBATCH --ntasks=1
#SBATCH --mem=10GB
#SBATCH --time=3:00:00
#SBATCH --job-name=minimal_VQE_cirq
#SBATCH --array=1-160
#SBATCH --output=minimal_VQE_cirq.log

module load nixpkgs/16.09 gcc/7.3.0 julia

julia --project -O3 --check-bounds=no run_graham.jl $SLURM_ARRAY_TASK_ID
