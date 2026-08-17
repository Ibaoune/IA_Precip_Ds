#!/bin/bash
#SBATCH --job-name=inf_u32
#SBATCH --output=logs/out_inf_u32_%j.log
#SBATCH --error=logs/out_inf_u32_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

echo "Starting UNet 32 LMDZ Inference..."
python3 -u ../src/predict.py config_unet_exp32_parallel_1979_2014.yaml
