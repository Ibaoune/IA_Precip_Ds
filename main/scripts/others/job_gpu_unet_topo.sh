#!/bin/bash
#SBATCH --job-name=unet_topo_gpu
#SBATCH --output=unet_topo_gpu%j.log
#SBATCH --error=unet_topo_gpu%j.log
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

CONFIG="../configs/unet/tests/config_unet_topo.yaml"

if [[ "$train" == "yes" ]]; then
    python3 -u ../train.py "$CONFIG"
fi
if [[ "$validation" == "yes" ]]; then
    python3 -u ../eval.py "$CONFIG"
fi
