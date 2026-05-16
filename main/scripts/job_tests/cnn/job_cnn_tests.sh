#!/bin/bash

#SBATCH --job-name=cnn_tests
#SBATCH --output=cnn_tests_%j.log
#SBATCH --error=cnn_tests_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

# Activate Conda environment
source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

TESTS_DIR="../../../configs/cnn/tests"

echo "Starting CNN tests from $TESTS_DIR"

for config in "$TESTS_DIR"/*.yaml; do
    echo "======================================"
    echo "Running test for config: $config"
    echo "======================================"
    
    echo "[INFO] Running training..."
    python3 -u ../../../train.py "$config"
    
    echo "[INFO] Running validation..."
    python3 -u ../../../eval.py "$config"
    
    echo "======================================"
done
