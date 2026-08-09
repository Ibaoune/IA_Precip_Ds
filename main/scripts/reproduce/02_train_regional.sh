#!/bin/bash
# ==============================================================================
# Script: 02_train_regional.sh
# Description: Generates the regional configurations (loss masking) and submits 
#              the training jobs for the 4 retained models over the 4 sub-regions.
# ==============================================================================

set -e

# Change to the root of the project
cd "$(dirname "$0")/../../.."

echo "=== Generating and Submitting Regional Training Jobs ==="

cat << 'EOF' > main/scripts/reproduce/generate_regional.py
import os
import yaml
import subprocess

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    configs_dir = os.path.join(base_dir, "main", "configs", "reproduce")
    jobs_dir = os.path.join(base_dir, "main", "scripts", "jobs", "reproduce", "regional")
    logs_dir = os.path.join(base_dir, "main", "scripts", "logs", "reproduce", "regional")
    shapefiles_dir = os.path.join(base_dir, "data", "shape_files")

    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    models = ["cnn", "unet", "vit", "glm"]
    regions = {
        "north": os.path.join(shapefiles_dir, "north.shp"),
        "north_east": os.path.join(shapefiles_dir, "north_east.shp"),
        "east": os.path.join(shapefiles_dir, "east.shp"),
        "south": os.path.join(shapefiles_dir, "south.shp")
    }

    target_dir = os.path.join(configs_dir, "regional")
    os.makedirs(target_dir, exist_ok=True)

    for model in models:
        base_yaml = os.path.join(configs_dir, "base", f"config_{model}.yaml")
        
        if not os.path.exists(base_yaml):
            print(f"[WARNING] Base config not found: {base_yaml}")
            continue

        with open(base_yaml, "r") as f:
            base_config = yaml.safe_load(f)

        for reg_name, shape_path in regions.items():
            reg_config = yaml.safe_load(yaml.dump(base_config))
            
            if "training" not in reg_config:
                reg_config["training"] = {}
                
            reg_config["training"]["loss_mask"] = {
                "enable": True,
                "region": reg_name,
                "shapefile": shape_path
            }
            
            # Reroute results to isolated reproduce directory
            reg_config["paths"]["results_dir"] = os.path.join(base_dir, "main", "results", "reproduce", "regional", f"{model}_{reg_name}")

            yaml_path = os.path.join(target_dir, f"config_{model}_{reg_name}.yaml")
            with open(yaml_path, "w") as out_f:
                yaml.dump(reg_config, out_f, default_flow_style=False)

            # Generate SLURM Script
            job_script = os.path.join(jobs_dir, f"job_gpu_{model}_regional_{reg_name}.sh")
            content = f"""#!/bin/bash
#SBATCH --job-name={model}_reg_{reg_name}
#SBATCH --output={logs_dir}/{model}_reg_{reg_name}_%j.out
#SBATCH --error={logs_dir}/{model}_reg_{reg_name}_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

source ~/.bashrc
conda activate clean_env_Pytorch
export PYTHONUNBUFFERED=1

cd {os.path.join(base_dir, "main")}
echo "Starting regional training for {model} on {reg_name}"
python3 -u train.py {yaml_path}
python3 -u eval.py {yaml_path}
"""
            with open(job_script, "w") as jf:
                jf.write(content)
            
            os.chmod(job_script, 0o755)
            print(f"Submitting {job_script}")
            subprocess.run(["sbatch", job_script])

if __name__ == "__main__":
    main()
EOF

python3 main/scripts/reproduce/generate_regional.py

echo "=== Regional Training Jobs Submitted ==="
echo "Check status with: squeue -u \$USER"
