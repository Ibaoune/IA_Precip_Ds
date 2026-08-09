#!/bin/bash
# ==============================================================================
# Script: 05_evaluate_lmdz.sh
# Description: Generates the post-processing configuration for evaluating the 
#              LMDZ inference outputs from the reproduce directories, and submits.
# ==============================================================================

set -e

cd "$(dirname "$0")/../../.."

echo "=== Generating LMDZ Post-processing configuration ==="

cat << 'EOF' > postproc/scripts/reproduce/generate_lmdz_postproc.py
import os
import yaml

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    out_config_path = os.path.join(base_dir, "postproc", "configs", "reproduce", "config_reproduce_lmdz.yaml")
    jobs_dir = os.path.join(base_dir, "postproc", "scripts", "jobs", "reproduce")
    logs_dir = os.path.join(base_dir, "postproc", "logs", "reproduce")
    
    os.makedirs(os.path.dirname(out_config_path), exist_ok=True)
    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    regions = ["north", "north_east", "south", "east"]
    models = ["cnn", "unet", "vit", "glm"]

    # We use MSWEP as reference
    config = {
        "experiment": "Reproduce_LMDZ_Evaluation",
        "parameters": {
            "regions": ["allmorr"] + regions,
            "plot_only": False,
            "impose_robust_limits": True,
            "show_title_metadata": True,
            "customize_colorbars": True,
            "plot_periods": ["Annual", "DJF", "JJA"]
        },
        "reference": {
            "name": "mswep",
            "file_path": "/srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/obs/mswep/mswep_1979_2020.nc",
            "variable_name": "precipitation"
        },
        "postproc": {
            "mean": {"enable": True, "bias": True, "rmse": True, "correlation": True},
            "extreme": {
                "enable": True, "cdd": True, "cdd_thresholds": [1.0, 0.5],
                "r95": True, "r95_freq": True, "qqplot": False, "r01": False, "r99": False, "r99_freq": False, "rocss": False
            }
        },
        "datasets": []
    }

    # Discover datasets from reproduce folders
    for model in models:
        # Global
        ds_global = {
            "name": f"{model.upper()}_Global",
            "file_path": os.path.join(base_dir, "inference", "results", "reproduce", "lmdz", f"{model}_global", f"{model}_global_lmdz_250_present_true.nc"),
            "variable_name": "precip",
            "test_start_date": "1979-01-01",
            "test_end_date": "2014-12-31"
        }
        if model == "glm":
            ds_global["name"] = "GLM"
        config["datasets"].append(ds_global)

        # Regionals (skip GLM)
        if model != "glm":
            for reg in regions:
                ds_reg = {
                    "name": f"{model.upper()}_{reg.capitalize()}",
                    "file_path": os.path.join(base_dir, "inference", "results", "reproduce", "lmdz", f"{model}_{reg}", f"{model}_{reg}_lmdz_250_present_true.nc"),
                    "variable_name": "precip",
                    "test_start_date": "1979-01-01",
                    "test_end_date": "2014-12-31"
                }
                config["datasets"].append(ds_reg)

    # Add LMDZ raw output for comparison
    config["datasets"].append({
        "name": "LMDZ250",
        "file_path": "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/data/lmdz/r250/amip/present/all_Mor/precip-hist.nc",
        "variable_name": "precip",
        "test_start_date": "1979-01-01",
        "test_end_date": "2014-12-31"
    })

    config["visualisation"] = {
        "model_colors": {
            "MSWEP": "#000000", "GLM": "#2A9D8F", "LMDZ250": "#FF8C00",
            "CNN_Global": "#1D3557", "CNN_North": "#457B9D", "CNN_South": "#A8DADC", "CNN_East": "#8ECAE6", "CNN_North_east": "#98C1D9",
            "UNET_Global": "#4A0082", "UNET_North": "#8A2BE2", "UNET_South": "#BBA0CA", "UNET_East": "#D8BFD8", "UNET_North_east": "#E6E6FA",
            "VIT_Global": "#9E0000", "VIT_North": "#E63946", "VIT_South": "#F4A261", "VIT_East": "#E9C46A", "VIT_North_east": "#F4A261"
        }
    }

    with open(out_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
    print(f"Generated {out_config_path}")

    # Generate SLURM Script
    job_script = os.path.join(jobs_dir, "job_evaluate_lmdz.sh")
    content = f"""#!/bin/bash
#SBATCH --job-name=eval_lmdz
#SBATCH --output={logs_dir}/eval_lmdz_%j.out
#SBATCH --error={logs_dir}/eval_lmdz_%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G

source ~/.bashrc
conda activate clean_env_Pytorch
export PYTHONUNBUFFERED=1

cd {os.path.join(base_dir, "postproc")}
echo "Running post-processing evaluation for LMDZ (1979-2014)"
python3 -u src/postproc.py configs/reproduce/config_reproduce_lmdz.yaml
"""
    with open(job_script, "w") as f:
        f.write(content)
    os.chmod(job_script, 0o755)

if __name__ == "__main__":
    main()
EOF

python3 postproc/scripts/reproduce/generate_lmdz_postproc.py

echo "=== Submitting LMDZ Post-processing Job ==="
sbatch postproc/scripts/jobs/reproduce/job_evaluate_lmdz.sh

echo "======================================================================"
echo " LMDZ Post-processing job submitted!"
echo " Check status with: squeue -u \$USER"
echo ""
echo " IMPORTANT: Once the job is FINISHED, run the summary generation."
echo "======================================================================"
