#!/bin/bash
#SBATCH --partition=compute
#SBATCH --job-name=DIST_SHIFT_FAM
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
RESULT_DIR="$SLURM_SUBMIT_DIR/results/distribution_shift_family"

mkdir -p "$SCRATCH_DIR"
mkdir -p "$RESULT_DIR"

# Copy experiment and shared utilities.
cp experiments/02_distribution_shift.py "$SCRATCH_DIR/"
cp experiments/common.py "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"

python3 02_distribution_shift.py \
    --datasets breast_cancer wine diabetes \
    --seeds "$SEED" \
    --family-check \
    --bins 3 \
    --epochs 500 \
    --output "distribution_shift_family_seed_${SEED}.csv"

cp "distribution_shift_family_seed_${SEED}.csv" "$RESULT_DIR/"

rm -rf "$SCRATCH_DIR"