#!/bin/bash

#SBATCH --job-name=postproc_vit_cnn
#SBATCH --output=postproc_vit_cnn_%j.log
#SBATCH --error=postproc_vit_cnn_%j.log
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

# Activate Conda environment
source ~/.bashrc
conda activate clean_env_Pytorch

start_time=$(date +%s)

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Comparing ViT tests with CNN vs MSWEP"
echo "======================================"

CONFIG="config_vit_vs_cnn.yaml"
python3 -u postproc.py "$CONFIG"

end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "======================================"
echo "Job ${SLURM_JOB_ID} completed in $runtime seconds."
echo "End time: $(date)"
echo "======================================"
