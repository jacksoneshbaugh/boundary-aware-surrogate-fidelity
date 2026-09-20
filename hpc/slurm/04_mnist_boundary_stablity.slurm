#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=MNIST_STAB
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=00-02:00:00
#SBATCH --output=slurm.%x.%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source hpc_venv/bin/activate
module load cuda

SEED="$SLURM_ARRAY_TASK_ID"

SCRATCH_DIR="/scratch/$USER/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}-${SEED}"
RESULT_DIR="$SLURM_SUBMIT_DIR/results/deep_boundary_stability/cache"

mkdir -p "$SCRATCH_DIR"
mkdir -p "$RESULT_DIR"

cp experiments/04_deep_boundary_stability.py "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"

python3 04_deep_boundary_stability.py \
    --dataset mnist \
    --seeds "$SEED" \
    --epochs 30 \
    --data-dir data \
    --output-dir output \
    --batch-size 128 \
    --num-workers 4

cp "output/mnist_seed_${SEED}.npz" "$RESULT_DIR/"

rm -rf "$SCRATCH_DIR"