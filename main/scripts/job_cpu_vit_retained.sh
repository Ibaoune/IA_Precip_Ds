#!/bin/bash
#SBATCH --job-name=vit_retained_cpu
#SBATCH --output=vit_retained_cpu%j.log
#SBATCH --error=vit_retained_cpu%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

train="yes"
validation="yes"
source ~/.bashrc
conda activate clean_env_Pytorch
export PYTHONUNBUFFERED=1

CONFIG="../configs/vit/retained/config.yaml"

if [[ "$train" == "yes" ]]; then
    python3 -u ../train.py "$CONFIG"
fi
if [[ "$validation" == "yes" ]]; then
    python3 -u ../eval.py "$CONFIG"
fi
