#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)
#SBATCH --job-name=plot_lmdz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=scripts/logs/plot_lmdz_%j.log
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

cd /srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc

echo "Starting SLURM Job: plot_lmdz_figures"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Directory: $(pwd)"
echo "=========================================================="

# Activate Conda Environment
source ~/.bashrc
conda activate clean_env_Pytorch

# Execute the python script
python3 -u src/plot_lmdz_paper_figures.py

echo "=========================================================="
echo "Job Completed Successfully!"
echo "Date: $(date)"
echo "=========================================================="
