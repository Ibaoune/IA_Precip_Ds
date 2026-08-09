#!/bin/bash
#SBATCH --job-name=unet_exp_test
#SBATCH --output=unet_exp_test_%j.log
#SBATCH --error=unet_exp_test_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G

source ~/.bashrc
conda activate clean_env_Pytorch
export PYTHONUNBUFFERED=1

# Change this if you want to test one specific variant
CONFIGS=(
    "../../configs/unet/tests/tests_advanced_arch/config_unet_exp28_strict_cpu_test.yaml"
    "../../configs/unet/tests/tests_advanced_arch/config_unet_exp29_dense_cpu_test.yaml"
    "../../configs/unet/tests/tests_advanced_arch/config_unet_exp30_residual_cpu_test.yaml"
    "../../configs/unet/tests/tests_advanced_arch/config_unet_exp31_attention_cpu_test.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
    echo "================================================="
    echo "Testing: $CONFIG"
    echo "================================================="
    python3 -u ../../train.py "$CONFIG"
done
