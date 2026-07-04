#!/bin/bash

# runs files on slurm clusters using modern python module resolution
# Usage: ./run.sh <mem> <time> <cpus> <python_file>
# Example: ./run.sh 48G 12:00:00 24 ../src/hydro_mode_tda_dce/run_h0_crocker_l2_norm.py

REPO=$(git rev-parse --show-toplevel)

# gets the absolute path of the file
ABS_PATH=$(realpath "$4")

# strip off the REPO path prefix and the .py extension to isolate the internal package path
RELATIVE_PATH=${ABS_PATH#"$REPO/"}
MODULE_PATH=${RELATIVE_PATH%.py}

# convert slashes to dots for python -m (e.g., src/test -> src.test)
MODULE_NAME=$(echo "$MODULE_PATH" | tr '/' '.')

sbatch \
    --job-name="$(basename "$4" .py)" \
    --account=physics --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task="$3" --mem="$1" --time="$2" <<EOF
#!/bin/bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source "$REPO/.venv/bin/activate"

# Change directory to the repository root so python resolves the package tree cleanly
cd "$REPO"

# Run as a module using the translated dotted string
python -m $MODULE_NAME
EOF