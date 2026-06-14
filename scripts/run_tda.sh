#!/bin/bash
#SBATCH --job-name=tda_multi_species
#SBATCH --account=physics       
#SBATCH --partition=cpu         
#SBATCH --qos=normal          
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50     
#SBATCH --mem=200G             
#SBATCH --time=12:00:00  

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO=$(git rev-parse --show-toplevel)
python "$REPO/src/tda.py"