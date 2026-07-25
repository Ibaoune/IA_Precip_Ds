#!/bin/bash
#SBATCH --job-name=cnn_retained_cpu
#SBATCH --output=cnn_retained_cpu%j.log
#SBATCH --error=cnn_retained_cpu%j.log
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

CONFIG="../configs/cnn/retained/config.yaml"

if [[ "$train" == "yes" ]]; then
    python3 -u ../train.py "$CONFIG"
fi
if [[ "$validation" == "yes" ]]; then
    python3 -u ../eval.py "$CONFIG"
fi
