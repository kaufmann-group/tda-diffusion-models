#!/bin/bash
#SBATCH --job-name=ipengines_tda
#SBATCH --account=physics
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=ipengines_%j.out
#SBATCH --error=ipengines_%j.err

echo "Starting IPyParallel engines"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Number of tasks / engines: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Working directory: $(pwd)"

REPO=$(git rev-parse --show-toplevel)

cd "$REPO" || exit 1
source .venv/bin/activate

echo "Repository: $REPO"
echo "Python: $(which python)"
echo "ipengine: $(which ipengine)"

srun ipengine --profile=default --log-to-file --log-level=20