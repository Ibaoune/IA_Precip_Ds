# Author: M. El Aabaribaoune (@um6p)
import os
import re

base_config_dir = "../../../configs/unet/tests/"
sweep_config_dir = "../../../configs/unet/tests/sweep/"
sweep_scripts_dir = "sweep_scripts/"

batch_sizes = [64, 128, 512, 1024]
learning_rates = ["1e-3", "1e-4", "1e-5"]

submit_all_script = open("submit_sweep.sh", "w")
submit_all_script.write("#!/bin/bash\n\ncd sweep_scripts\n")

for exp_num in range(1, 7):
    base_file = os.path.join(base_config_dir, f"test_exp{exp_num}.yaml")
    with open(base_file, "r") as f:
        content = f.read()
    
    # Replace dates
    content = re.sub(r"start:\s*'1979-01-01'", "start: '1979-01-01'", content)
    content = re.sub(r"end:\s*'1983-12-31'", "end: '2005-12-31'", content)
    content = re.sub(r"start:\s*'1984-01-01'", "start: '2006-01-01'", content)
    content = re.sub(r"end:\s*'1984-12-31'", "end: '2020-12-31'", content)

    for bs in batch_sizes:
        for lr in learning_rates:
            new_content = content
            
            # Replace batch size
            new_content = re.sub(r"batch_size:\s*\d+", f"batch_size: {bs}", new_content)
            
            # Replace learning rate
            new_content = re.sub(r"learning_rate:\s*[\d.e-]+", f"learning_rate: {lr}", new_content)
            
            # Replace experiment name
            match = re.search(r"experiment:\s*(\S+)", new_content)
            if match:
                base_exp_name = match.group(1)
                new_exp_name = f"{base_exp_name}_bs{bs}_lr{lr}"
                new_content = re.sub(r"experiment:\s*(\S+)", f"experiment: {new_exp_name}", new_content)
            else:
                new_exp_name = f"unet_exp{exp_num}_bs{bs}_lr{lr}"
            
            # Save new config
            new_config_filename = f"test_exp{exp_num}_bs{bs}_lr{lr}.yaml"
            new_config_path = os.path.join(sweep_config_dir, new_config_filename)
            with open(new_config_path, "w") as f:
                f.write(new_content)
            
            # Generate slurm script
            script_filename = f"run_exp{exp_num}_bs{bs}_lr{lr}.sh"
            script_path = os.path.join(sweep_scripts_dir, script_filename)
            
            slurm_content = f"""#!/bin/bash
#SBATCH --job-name=e{exp_num}_b{bs}_l{lr}
#SBATCH --output=../logs/sweep_exp{exp_num}_bs{bs}_lr{lr}_%j.log
#SBATCH --error=../logs/sweep_exp{exp_num}_bs{bs}_lr{lr}_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

echo "======================================"
echo "Job ID: \\$SLURM_JOB_ID"
echo "Running Sweep: Exp {exp_num}, BS {bs}, LR {lr}"
echo "======================================"

cd ../../../..
python3 -u train.py configs/unet/tests/sweep/{new_config_filename}
python3 -u eval.py configs/unet/tests/sweep/{new_config_filename}

echo "======================================"
echo "Job completed."
echo "======================================"
"""
            with open(script_path, "w") as f:
                f.write(slurm_content)
            
            submit_all_script.write(f"sbatch {script_filename}\n")

submit_all_script.write("cd ..\n")
submit_all_script.close()
