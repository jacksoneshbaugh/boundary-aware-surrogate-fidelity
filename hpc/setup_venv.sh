#!/bin/bash
# ============================================
# HPC Virtual Environment Setup Script
# Run this on the login node before submitting jobs.
# Usage: bash setup_venv.sh
# ============================================

VENV_DIR="hpc_venv"

echo "=== HPC Venv Setup ==="

# Delete old venv if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Removing old virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create fresh venv
echo "Creating new virtual environment..."
python3 -m venv "$VENV_DIR"

# Activate and install
echo "Installing dependencies..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install torch torchvision numpy scikit-learn scipy

echo ""
echo "=== Setup Complete ==="
echo "Installed packages:"
pip list --format=columns | grep -iE "torch|numpy|scikit|scipy"
echo ""
echo "You can now submit SLURM jobs."
