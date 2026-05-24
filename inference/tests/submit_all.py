# Author: M. El Aabaribaoune (@um6p)
import os
import subprocess

configs = ["config_cnn_exp3.yaml", "config_cnn_exp5.yaml", "config_vit_exp2.yaml", "config_glm_precip_l2.yaml"]

for config in configs:
 job_name = config.replace(".yaml", "")
 sh_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=out_{job_name}_%j.log
#SBATCH --error=out_{job_name}_%j.log
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

echo "Running {config}..."
python3 -u predict.py {config}
"""
 sh_file = f"run_{job_name}.sh"
 with open(sh_file, "w") as f:
 f.write(sh_content)
 
 print(f"Submitting {sh_file}...")
 subprocess.run(["sbatch", sh_file])
