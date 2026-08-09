#!/bin/bash
#SBATCH --job-name=cnn_retained_gpu
#SBATCH --output=cnn_retained_gpu%j.log
#SBATCH --error=cnn_retained_gpu%j.log
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

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
