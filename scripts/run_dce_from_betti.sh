#!/bin/bash
#SBATCH --job-name=tda_multi_species
#SBATCH --account=physics       
#SBATCH --partition=cpu         
#SBATCH --qos=standby         # changed temorarly since `squeue -A physics` is busy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50    
#SBATCH --mem=200G            
#SBATCH --time=04:00:00      # max allowed for standby

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO=$(git rev-parse --show-toplevel)
python "$REPO/src/dce_from_betti.py"