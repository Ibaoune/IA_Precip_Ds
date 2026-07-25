import os

slurm_template = """#!/bin/bash
#SBATCH --job-name=eval_{exp_name}
#SBATCH --output=logs/eval_{exp_name}_%j.log
#SBATCH --error=logs/eval_{exp_name}_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running Evaluation for: {exp_name}"
echo "======================================"

cd ../..
python3 -u eval.py {config_path}

echo "======================================"
echo "Job completed."
echo "======================================"
"""

os.makedirs("scripts/job_eval", exist_ok=True)
os.makedirs("logs", exist_ok=True)

configs_to_evaluate = []

# 1. New experiments
for i in range(1, 11):
    configs_to_evaluate.append((f"unet_exp{i}", f"configs/experiments/unet_exp{i}.yaml"))
for i in range(1, 9):
    configs_to_evaluate.append((f"vit_exp{i}", f"configs/experiments/vit_exp{i}.yaml"))

# 2. Retained models
configs_to_evaluate.append(("unet_retained", "configs/unet/retained/config.yaml"))
configs_to_evaluate.append(("vit_retained", "configs/vit/retained/config.yaml"))
configs_to_evaluate.append(("cnn_retained", "configs/cnn/retained/config.yaml"))

generated_scripts = []

for exp_name, config_path in configs_to_evaluate:
    script_content = slurm_template.format(exp_name=exp_name, config_path=config_path)
    script_path = f"scripts/job_eval/run_eval_{exp_name}.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    generated_scripts.append(script_path)

# Generate a master script to submit them all
with open("scripts/job_eval/submit_all_eval.sh", "w") as f:
    f.write("#!/bin/bash\n")
    for s in generated_scripts:
        f.write(f"sbatch {os.path.basename(s)}\n")
        
os.chmod("scripts/job_eval/submit_all_eval.sh", 0o755)
print(f"✅ Generated {len(generated_scripts)} evaluation scripts.")
print("✅ Generated master submit script: scripts/job_eval/submit_all_eval.sh")
