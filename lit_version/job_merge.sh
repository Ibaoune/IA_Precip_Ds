#!/bin/bash

#SBATCH --job-name=emul_cpu        # Job Name
#SBATCH --output=out_%j.log        # Everything (stdout & stderr) goes here
#SBATCH --error=out_%j.log         # Can also merge stderr with stdout
#SBATCH --nodes=1
#SBATCH --time=36:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

# Activate Conda environment
source ~/.bashrc
conda activate clean_env_Pytorch

# Record start time
start_time=$(date +%s)

# Run main script
# stdout and stderr are automatically redirected to the SLURM output file
#python3 -u train.py  # -u = unbuffered output for real-time logging

echo "Running merge.."
python3 -u  merge.py

# Record end time and log total runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Job ${SLURM_JOB_ID} completed in $runtime seconds."

