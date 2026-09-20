#!/bin/bash
#SBATCH --partition=compute
#SBATCH --job-name=CF_TRANSFER
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
RESULT_DIR="$SLURM_SUBMIT_DIR/results/counterfactual_transfer"

mkdir -p "$SCRATCH_DIR"
mkdir -p "$RESULT_DIR"

# Copy experiment and shared utilities to scratch.
cp experiments/03_counterfactual_transfer.py "$SCRATCH_DIR/"
cp experiments/common.py "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"

python3 03_counterfactual_transfer.py \
    --datasets breast_cancer wine diabetes \
    --seeds "$SEED" \
    --depths 2 3 5 7 10 \
    --epochs 500 \
    --boundary-fraction 0.25 \
    --output "counterfactual_transfer_seed_${SEED}.csv" \
    --query-output "counterfactual_transfer_queries_seed_${SEED}.csv"

# Copy canonical outputs back to the repository.
cp "counterfactual_transfer_seed_${SEED}.csv" "$RESULT_DIR/"
cp "counterfactual_transfer_queries_seed_${SEED}.csv" "$RESULT_DIR/"

rm -rf "$SCRATCH_DIR"