#!/bin/bash
# Usage: ./script.sh <mem> <time> <cpus> <python_file>

REPO=$(git rev-parse --show-toplevel)
PYTHON_FILE=$(realpath "$4")

sbatch \
    --job-name="$(basename "$4" .py)" \
    --account=physics --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task="$3" --mem="$1" --time="$2" <<EOF
#!/bin/bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source "$REPO/.venv/bin/activate"
python "$PYTHON_FILE"
EOF