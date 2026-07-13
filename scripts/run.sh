#!/bin/bash

# Run a Python module through SLURM, optionally with an IPyParallel cluster.
# usage:
#   ./run.sh <mem> <time> <cpus> <python_file>
#   ./run.sh 32G 168:00:00 38 ../src/hydro_mode_tda_dce/h1_beta_curve_area_four_regimes.py --start-ipyengines

set -e

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "Usage: $0 <mem> <time> <cpus> <python_file> [--start-ipyengines]"
    exit 1
fi

MEMORY="$1"
WALLTIME="$2"
CPUS="$3"
PYTHON_FILE="$4"
START_IPYENGINES=false

if [ "$#" -eq 5 ]; then
    if [ "$5" != "--start-ipyengines" ]; then
        echo "Unknown option: $5"
        echo "Valid option: --start-ipyengines"
        exit 1
    fi

    START_IPYENGINES=true
fi

REPO=$(git rev-parse --show-toplevel)
ABS_PATH=$(realpath "$PYTHON_FILE")
RELATIVE_PATH=${ABS_PATH#"$REPO/"}
MODULE_PATH=${RELATIVE_PATH%.py}
MODULE_NAME=$(echo "$MODULE_PATH" | tr '/' '.')
JOB_NAME=$(basename "$PYTHON_FILE" .py)

sbatch \
    --job-name="$JOB_NAME" \
    --account=physics \
    --partition=cpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEMORY" \
    --time="$WALLTIME" <<EOF
#!/bin/bash

set -e

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REPO="$REPO"
MODULE_NAME="$MODULE_NAME"
JOB_NAME="$JOB_NAME"
N_ENGINES="$CPUS"
START_IPYENGINES="$START_IPYENGINES"

cd "\$REPO"
module load conda
source activate "$REPO/.venv"
which python

echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_JOB_NODELIST"
echo "Repository: \$REPO"
echo "Python: \$(which python)"
echo "Module: \$MODULE_NAME"
echo "Allocated CPUs: \$SLURM_CPUS_PER_TASK"
echo "Start IPyParallel: \$START_IPYENGINES"

CONTROLLER_PID=""
ENGINE_PIDS=()

cleanup() {
    if [ "\$START_IPYENGINES" = true ]; then
        echo "Stopping IPyParallel processes..."

        for pid in "\${ENGINE_PIDS[@]}"; do
            kill "\$pid" 2>/dev/null || true
        done

        if [ -n "\$CONTROLLER_PID" ]; then
            kill "\$CONTROLLER_PID" 2>/dev/null || true
        fi

        wait 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

if [ "\$START_IPYENGINES" = true ]; then
    export IPP_PROFILE="slurm_\${SLURM_JOB_ID}"
    export IPYTHONDIR="\$REPO/.ipython"

    echo "IPyParallel profile: \$IPP_PROFILE"
    echo "Starting IPyParallel controller..."

    IPP_LOG_DIR="\$IPYTHONDIR/profile_\$IPP_PROFILE/log"
    mkdir -p "\$IPP_LOG_DIR"

    ipcontroller --profile="\$IPP_PROFILE" --ip="127.0.0.1" --log-to-file --log-level=30 >"\$IPP_LOG_DIR/controller_startup.log" 2>&1 &
    CONTROLLER_PID=\$!

    CONNECTION_FILE="\$IPYTHONDIR/profile_\$IPP_PROFILE/security/ipcontroller-client.json"

    echo "Waiting for controller connection file..."

    for attempt in \$(seq 1 60); do
        if [ -f "\$CONNECTION_FILE" ]; then
            break
        fi

        if ! kill -0 "\$CONTROLLER_PID" 2>/dev/null; then
            echo "IPyParallel controller exited before creating its connection file."
            exit 1
        fi

        sleep 1
    done

    if [ ! -f "\$CONNECTION_FILE" ]; then
        echo "Controller connection file was not created:"
        echo "  \$CONNECTION_FILE"
        exit 1
    fi

    echo "Starting \$N_ENGINES IPyParallel engines..."

    for engine_index in \$(seq 1 "\$N_ENGINES"); do
        ipengine --profile="\$IPP_PROFILE" --log-to-file --log-level=30 >"\$IPP_LOG_DIR/engine_\${engine_index}_startup.log" 2>&1 &
        ENGINE_PIDS+=("\$!")
    done

    echo "Waiting for engines to connect..."

    python - <<'PYTHON'
import os
import time
import ipyparallel as ipp

profile = os.environ["IPP_PROFILE"]
expected_engines = int(os.environ["SLURM_CPUS_PER_TASK"])
connected = 0

for attempt in range(300):
    try:
        rc = ipp.Client(profile=profile)
        connected = len(rc.ids)
        print(f"Connected engines: {connected}/{expected_engines}", flush=True)
        rc.close()

        if connected >= expected_engines:
            break
    except Exception as error:
        print(f"Waiting for IPyParallel cluster: {error}", flush=True)

    time.sleep(1)
else:
    raise RuntimeError(f"Only {connected}/{expected_engines} engines connected.")
PYTHON
fi

export TQDM_DISABLE=1

echo "Running module: \$MODULE_NAME"
python -m cProfile -o "$REPO/scripts/\${JOB_NAME}.prof" -m "\${MODULE_NAME}"
EOF