# Author: M. El Aabaribaoune (@um6p)
import os

template = """#!/bin/bash
#SBATCH --job-name=unet_{exp}
#SBATCH --output=logs/unet_{exp}_%j.log
#SBATCH --error=logs/unet_{exp}_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running UNet Experiment: {exp}"
echo "======================================"

cd ../..
python3 -u train.py configs/unet/tests/test_exp{exp}.yaml

echo "======================================"
echo "Job completed."
echo "======================================"
"""

os.makedirs("main/scripts/job_tests", exist_ok=True)
for i in range(1, 13):
    script_content = template.format(exp=i)
    script_path = f"main/scripts/job_tests/run_test_unet_exp{i}.sh"
    with open(script_path, "w") as f:
        f.write(script_content)

print("Generated 5 SLURM scripts.")
