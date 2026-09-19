#!/bin/bash
#SBATCH --partition=compute
#SBATCH --job-name=BOUNDARY_FID
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=00-02:00:00
#SBATCH --output=slurm.%x.%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source hpc_venv/bin/activate

SEED="$SLURM_ARRAY_TASK_ID"

SCRATCH_DIR="/scratch/$USER/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}-${SEED}"
RESULT_DIR="$SLURM_SUBMIT_DIR/results/boundary_fidelity"

mkdir -p "$SCRATCH_DIR"
mkdir -p "$RESULT_DIR"

# Copy the experiment and its shared utilities.
cp experiments/01_boundary_fidelity.py "$SCRATCH_DIR/"
cp experiments/common.py "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"

python3 01_boundary_fidelity.py \
    --seeds "$SEED" \
    --depths 1 2 3 5 7 10 \
    --fractions 0.10 0.20 0.25 0.40 \
    --epochs 500 \
    --output "boundary_fidelity_seed_${SEED}.csv"

cp "boundary_fidelity_seed_${SEED}.csv" "$RESULT_DIR/"

rm -rf "$SCRATCH_DIR"