#!/bin/bash
#SBATCH --job-name=inf_u35
#SBATCH --output=logs/out_inf_u35_%j.log
#SBATCH --error=logs/out_inf_u35_%j.log
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU
#SBATCH --partition=compute

source ~/.bashrc
conda activate clean_env_Pytorch

echo "Starting UNet 35 LMDZ Inference..." 
cd /srv/data/mohammad.elaabaribao/work/papers/downscaling/inference/src
python3 -u predict.py ../tests/config_unet_exp35_1x1_hybrid_1979_2014.yaml
