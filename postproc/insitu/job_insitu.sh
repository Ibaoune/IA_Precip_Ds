#!/bin/bash
#SBATCH --job-name=insitu_plot
#SBATCH --output=out_annual_cycle_%j.log
#SBATCH --error=out_annual_cycle_%j.log
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

# Activate Conda environment
source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

echo "Starting annual cycle plotting script on CPU..."
python3 -u annual_cycle.py
echo "Script finished!"
