#!/bin/bash
# ==============================================================================
# Script: 04_run_lmdz_inference.sh
# Description: Generates the inference configurations for all retained models 
#              (global + regional) to run on LMDZ250 predictors, reading from 
#              the isolated reproduce directories. Submits the jobs.
# ==============================================================================

set -e

cd "$(dirname "$0")/../../.."

echo "=== Generating LMDZ Inference Configurations ==="

cat << 'EOF' > inference/scripts/reproduce/generate_lmdz.py
import os
import yaml
import subprocess

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    out_configs_dir = os.path.join(base_dir, "inference", "configs", "reproduce", "lmdz")
    jobs_dir = os.path.join(base_dir, "inference", "scripts", "jobs", "reproduce")
    logs_dir = os.path.join(base_dir, "inference", "logs", "reproduce")

    os.makedirs(out_configs_dir, exist_ok=True)
    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    models_info = {
        "cnn": {"type": "cnn_exp5", "model_file": "cnn_exp5_precip.pth"},
        "unet": {"type": "unet_exp32", "model_file": "unet_exp32_precip.pth"},
        "vit": {"type": "vit", "model_file": "vit_precip.pth"},
        "glm": {"type": "glm", "model_file": "glm_precip.pth"}
    }

    regions = ["north", "north_east", "south", "east"]

    for model, info in models_info.items():
        # Global variant
        variants = [{"name": f"{model}_global", "path": os.path.join("main", "results", "reproduce", "global", model)}]
        
        # Regional variants (skip GLM)
        if model != "glm":
            for reg in regions:
                variants.append({
                    "name": f"{model}_{reg}", 
                    "path": os.path.join("main", "results", "reproduce", "regional", f"{model}_{reg}")
                })
            
        for variant in variants:
            var_name = variant["name"]
            abs_var_path = os.path.join(base_dir, variant["path"])
            
            train_config_path = os.path.join(abs_var_path, "config.txt")
            model_path = os.path.join(abs_var_path, "models", info["model_file"])

            if not os.path.exists(train_config_path):
                print(f"[WARNING] Skipping {var_name}, config not found: {train_config_path}")
                continue

            output_dir = f"results/reproduce/lmdz/{var_name}"

            # Create inference YAML
            cfg = {
                "general": {
                    "experiment": var_name,
                    "model_type": info["type"],
                    "verbose": True
                },
                "paths": {
                    "lmdz_predictor_pattern": "{folder}/{lmdz_var}-hist.nc"
                },
                "mappings": {
                    "lmdz_var_map": {"z": "geop", "q": "rhum", "u": "vitu", "v": "vitv", "t": "temp"}
                },
                "prediction": {
                    "train_config_path": train_config_path,
                    "model_path": model_path,
                    "output_dir": output_dir,
                    "scenarios": [
                        {
                            "name": "lmdz_250_present",
                            "enable": True,
                            "src": "lmdz",
                            "start": "1979-01-01",
                            "end": "2014-12-31",
                            "bias_correction": True,
                            "folder": "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/data/lmdz/r250/amip/present/all_Mor",
                            "bc_reference_folder": "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/data/lmdz/r250/amip/present/all_Mor"
                        }
                    ]
                }
            }

            yaml_path = os.path.join(out_configs_dir, f"config_{var_name}.yaml")
            with open(yaml_path, "w") as out_f:
                yaml.dump(cfg, out_f, default_flow_style=False, sort_keys=False)

            # Generate SLURM Script
            job_script = os.path.join(jobs_dir, f"job_infer_{var_name}.sh")
            content = f"""#!/bin/bash
#SBATCH --job-name=inf_{var_name}
#SBATCH --output={logs_dir}/inf_{var_name}_%j.out
#SBATCH --error={logs_dir}/inf_{var_name}_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

source ~/.bashrc
conda activate clean_env_Pytorch
export PYTHONUNBUFFERED=1

cd {os.path.join(base_dir, "inference")}
echo "Starting LMDZ Inference for {var_name}"
python3 -u src/predict.py configs/reproduce/lmdz/config_{var_name}.yaml
"""
            with open(job_script, "w") as jf:
                jf.write(content)
            
            os.chmod(job_script, 0o755)
            print(f"Submitting {job_script}")
            subprocess.run(["sbatch", job_script])

if __name__ == "__main__":
    main()
EOF

python3 inference/scripts/reproduce/generate_lmdz.py

echo "=== LMDZ Inference Jobs Submitted ==="
echo "Check status with: squeue -u \$USER"
