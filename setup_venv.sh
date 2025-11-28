#!/bin/bash
# Setup virtual environment with uv
# Run this once before submitting jobs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"

echo "Setting up virtual environment with uv..."
echo "Location: $VENV_DIR"

# Load Python module
module load python/3.10.9-fasrc01

# Check if uv is available, if not install it
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install --user uv
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv "$VENV_DIR" --python 3.10

# Activate and install packages
echo "Installing packages..."
source "$VENV_DIR/bin/activate"

# Install packages using requirements.txt if it exists, otherwise install individually
if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
    echo "Installing from requirements.txt..."
    uv pip install -r "$SCRIPT_DIR/requirements.txt"
else
    echo "Installing packages individually..."
    uv pip install "numpy<2.0" "scipy<1.11"
    uv pip install mujoco
    uv pip install "gymnasium[mujoco]"
    uv pip install stable-baselines3 pandas matplotlib tqdm rich pyyaml
fi

echo ""
echo "Virtual environment setup complete!"
echo "Location: $VENV_DIR"
echo ""
echo "To test locally:"
echo "  source venv/bin/activate"
echo "  python train_rl.py --task 'Hopper-v5' --algorithm 'SAC' --seed 0"

