#!/bin/bash
#SBATCH --job-name=tda_multi_species
#SBATCH --account=physics       
#SBATCH --partition=cpu         
#SBATCH --qos=normal          
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50     # Matches your N_runs=50 perfectly
#SBATCH --mem=200G             # Ensure ~4G per process, 200G is great
#SBATCH --time=2-00:00:00      # Increased to 48 hours to avoid mid-run timeouts

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO=$(git rev-parse --show-toplevel)
python "$REPO/src/dce_from_betti.py"