#!/bin/bash
#SBATCH --job-name=cnns_lmdz250
#SBATCH --output=logs/out_cnns_lmdz250_%j.log
#SBATCH --error=logs/out_cnns_lmdz250_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

echo "Removing corrupted CNN files..."
rm -f results/1979_2014/cnn_exp3/cnn_lmdz_250_present_true.nc
rm -f results/1979_2014/cnn_exp5/cnn_lmdz_250_present_true.nc

echo "Running CNN_Exp3 for LMDZ250..."
python3 -u predict.py config_cnn_exp3_1979_2014.yaml > logs/cnn_exp3_lmdz250.log 2>&1

echo "Running CNN_Exp5 for LMDZ250..."
python3 -u predict.py config_cnn_exp5_1979_2014.yaml > logs/cnn_exp5_lmdz250.log 2>&1

echo "CNN inferences completed."
