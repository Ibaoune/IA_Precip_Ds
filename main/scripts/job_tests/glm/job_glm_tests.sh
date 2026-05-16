#!/bin/bash

#SBATCH --job-name=glm_tests
#SBATCH --output=glm_tests_%j.log
#SBATCH --error=glm_tests_%j.log
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

# Activate Conda environment
source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

TESTS_DIR="../../../configs/glm/tests"

echo "Starting GLM tests from $TESTS_DIR"

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
