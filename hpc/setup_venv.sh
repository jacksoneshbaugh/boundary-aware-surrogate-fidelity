#!/bin/bash
# ============================================
# HPC Virtual Environment Setup
#
# Creates the repository's HPC-specific Python
# environment and installs experiment dependencies.
#
# Usage:
#   bash hpc/setup_venv.sh
#
# May be invoked from any working directory.
# ============================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$REPO_ROOT/hpc_venv"

echo "=== HPC Venv Setup ==="
echo "Repository: $REPO_ROOT"
echo "Environment: $VENV_DIR"

# Delete old venv if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Removing old virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create fresh venv
echo "Creating new virtual environment..."
python3 -m venv "$VENV_DIR"

# Activate and install dependencies
echo "Installing dependencies..."
source "$VENV_DIR/bin/activate"

python3 -m pip install --upgrade pip
python3 -m pip install -r "$REPO_ROOT/requirements.txt"

echo ""
echo "=== Setup Complete ==="
echo "Installed packages:"
python3 -m pip list --format=columns | grep -iE "torch|numpy|scikit|scipy"
echo ""
echo "You can now submit SLURM jobs."