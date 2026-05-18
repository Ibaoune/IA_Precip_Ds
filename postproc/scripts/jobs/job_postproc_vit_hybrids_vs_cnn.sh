#!/bin/bash
#SBATCH --job-name=postproc_vit_hybrids_vs_cnn
#SBATCH --output=scripts/logs/postproc_vit_hybrids_vs_cnn_%j.log
#SBATCH --error=scripts/logs/postproc_vit_hybrids_vs_cnn_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Post-processing ViT Hybrids vs CNN (LMDZ 250)"
echo "======================================"

python3 -u src/postproc.py configs/config_vit_hybrids_vs_cnn.yaml

echo "======================================"
echo "Job completed."
echo "======================================"
