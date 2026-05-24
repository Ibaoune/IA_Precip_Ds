import os
import subprocess

test_configs = [f"../../../configs/vit/tests/test_exp{i}.yaml" for i in range(11, 21)]

for config_path in test_configs:
    config_filename = os.path.basename(config_path)
    job_name = config_filename.replace(".yaml", "")
    
    sh_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}_%j.log
#SBATCH --error={job_name}_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

CONFIG_PATH="{config_path}"

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running test for config: $CONFIG_PATH"
echo "======================================"

echo "[INFO] Running training..."
python3 -u ../../../train.py "$CONFIG_PATH"

echo "[INFO] Running validation..."
python3 -u ../../../eval.py "$CONFIG_PATH"

echo "======================================"
echo "Job completed."
echo "======================================"
"""
    
    sh_file = f"run_{job_name}.sh"
    with open(sh_file, "w") as f:
        f.write(sh_content)
    
    print(f"Generated {sh_file}. Submitting to SLURM...")
    subprocess.run(["sbatch", sh_file])
