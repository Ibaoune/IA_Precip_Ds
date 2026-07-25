import os
import yaml

# Base config template based on CNN retained configuration
base_config = {
    "general": {
        "verbose": True,
        "variable": "precip",
        "target": "mswep",
        "src": "era5",
        "interpolation_type": "linear",
        "variables": ["z", "q", "t", "u", "v"],
        "levels": [500, 700, 850, 1000],
        "resolution": 2.0
    },
    "training": {
        "learning_rate": 1e-4,
        "epochs": 200,
        "batch_size": 64,
        "loss_type": "bernoulli_gamma",
        "norm_mode": "gridbox",
        "early_stopping": {"enable": True, "max": 15},
        "LR_scheduler": {"enable": False, "patience": 5, "factor": 0.5, "min_lr": 1e-6},
        "gradient_clipping": {"enable": True, "value": 1.0},
        "weight_decay": {"enable": True, "value": 1e-5},
        "optimizer": "adamw",
        "dropout": {"enable": True, "value": 0.2},
        "scheduler": {"enable": True, "type": "cosine"},
        "group_norm": {"enable": True, "num_groups": 32},
        "validation": {"enable": True, "pecentage": 0.2}
    },
    "model": {
        "vit": {
            "emb_size": 128,
            "patch_size": 4,
            "num_layers": 4,
            "num_heads": 4,
            "dropout": 0.0
        }
    },
    "region": {
        "lon_min": -18, "lon_max": 0, "lat_min": 21, "lat_max": 37
    },
    "dates": {
        "train": {"start": "1979-01-01", "end": "2005-12-31"},
        "test": {"start": "2006-01-01", "end": "2020-12-31"}
    },
    "plots": {
        "eval": {"show_suffix_components_in_title": True}
    },
    "paths": {
        "root_dir": "/srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/reanalysis/era5",
        "shapefile_path": "/srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/shapefiles/Morocco_shpfile/DA_REGIONS_12R.shp",
        "era5_predictor_pattern": "/srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/reanalysis/era5/{var}_1979-2020_levels.nc",
        "mswep_path": "/srv/data/mohammad.elaabaribao/work/papers/downscaling/lit_version/data/mswep_1979_2020.nc" # Updated local path
    }
}

slurm_template = """#!/bin/bash
#SBATCH --job-name={exp_name}
#SBATCH --output=logs/{exp_name}_%j.log
#SBATCH --error=logs/{exp_name}_%j.log
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
echo "Running Experiment: {exp_name}"
echo "======================================"

cd ../..
python3 -u train.py {config_path}

echo "======================================"
echo "Job completed."
echo "======================================"
"""

os.makedirs("configs/experiments", exist_ok=True)
os.makedirs("scripts/job_experiments", exist_ok=True)
os.makedirs("logs", exist_ok=True)

experiments = [f"unet_exp{i}" for i in range(1, 11)] + [f"vit_exp{i}" for i in range(1, 9)]

generated_scripts = []

for exp in experiments:
    # 1. Generate YAML
    config = eval(repr(base_config))  # deep copy
    config["general"]["model_type"] = exp
    config["general"]["experiment"] = exp
    config["paths"]["results_dir"] = f"/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/experiments/{exp}/"
    
    config_path = f"configs/experiments/{exp}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
    # 2. Generate SLURM Script
    script_content = slurm_template.format(exp_name=exp, config_path=config_path)
    script_path = f"scripts/job_experiments/run_{exp}.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
        
    generated_scripts.append(script_path)

print(f"✅ Generated {len(generated_scripts)} experiment configs and SLURM scripts.")

# 3. Generate a master script to submit them all
with open("scripts/job_experiments/submit_all.sh", "w") as f:
    f.write("#!/bin/bash\n")
    for s in generated_scripts:
        f.write(f"sbatch {os.path.basename(s)}\n")
        
os.chmod("scripts/job_experiments/submit_all.sh", 0o755)
print("✅ Generated master submit script: scripts/job_experiments/submit_all.sh")
