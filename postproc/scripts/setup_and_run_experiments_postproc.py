"""
Author: M. El Aabaribaoune (@um6p)
Description: Main orchestrator script for the post-processing and evaluation pipeline.
"""


import os
import glob
import yaml

def find_file(pattern, search_dir="../../main/results/"):
    matches = glob.glob(os.path.join(search_dir, "**", pattern), recursive=True)
    if matches:
        return matches[0]
    return None

def main():
    os.makedirs("../../configs/experiments", exist_ok=True)
    
    # Common postproc template base
    base_config = {
        "experiment": "",
        "parameters": {
            "start_date": "2006-01-01",
            "end_date": "2020-12-31",
            "region": "allmorr",
            "regions": ["allmorr"],
            "mask_land": True,
            "only_morocco": True,
            "predictand": "pr",
            "plot_only": False,
            "impose_robust_limits": True,
            "show_title_metadata": True,
            "customize_colorbars": True,
            "plot_periods": ["Annual", "DJF", "MAM", "JJA", "SON"]
        },
        "reference": {
            "name": "MSWEP",
            "file_path": "/srv/data/mohammad.elaabaribao/work/papers/downscaling/lit_version/data/mswep_1979_2020.nc",
            "variable_name": "precipitation"
        },
        "datasets": [],
        "postproc": {
            "mean": {"enable": True, "bias": True, "rmse": True, "correlation": True},
            "extreme": {
                "enable": True, "cdd": True, "cdd_thresholds": [1.0],
                "qqplot": False, "r01": False, "r99": False, "r95": True,
                "r99_freq": False, "r95_freq": True, "rocss": False
            }
        },
        "visualisation": {
            "model_colors": {
                "MSWEP": "#000000",
                "CNN_Base": "#457B9D"
            },
            "metrics": {
                "mean": {
                    "colormaps": {"spatial": "custom_precip", "difference": "custom_bias"},
                    "limits": {
                        "spatial": [0, 0.1, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                        "boxplot": [0, 6], "temporal": [0, 6]
                    }
                },
                "bias": {
                    "colormaps": {"spatial": "custom_bias", "difference": "custom_bias"},
                    "limits": {
                        "spatial": [-5, -4, -3, -2, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5],
                        "boxplot": [-2.5, 2.5], "temporal": [-2.5, 2.5]
                    }
                },
                "rmse": {
                    "colormaps": {"spatial": "custom_precip", "difference": "custom_bias"},
                    "limits": {
                        "spatial": [0, 0.1, 0.3, 0.5, 1, 1.5, 2],
                        "boxplot": [0, 2], "temporal": [0, 2]
                    }
                },
                "cdd": {
                    "colormaps": {"spatial": "YlOrBr", "difference": "custom_bias"},
                    "limits": {
                        "spatial": [0, 10, 20, 30, 40, 60, 80, 100, 150, 200, 250, 300, 400],
                        "difference": [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100],
                        "boxplot": [0, 400], "temporal": [0, 400]
                    }
                },
                "r95": {
                    "colormaps": {"spatial": "custom_precip", "difference": "custom_bias"},
                    "limits": {
                        "spatial": [0, 0.1, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10],
                        "difference": [-5, -4, -3, -2, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5],
                        "boxplot": [0, 10], "temporal": [0, 10]
                    }
                },
                "r95_freq": {
                    "colormaps": {"spatial": "custom_freq", "difference": "custom_bias"},
                    "limits": {
                        "spatial": [0, 2, 4, 6, 8, 10, 15],
                        "difference": [-10, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 10],
                        "boxplot": [0, 15], "temporal": [0, 15]
                    }
                },
                "correlation": {
                    "colormaps": {"spatial": "RdBu_r", "difference": "custom_bias"},
                    "limits": {
                        "spatial": [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
                        "boxplot": [-1, 1], "temporal": [-1, 1]
                    }
                },
                "default": {
                    "colormaps": {"spatial": "viridis", "difference": "custom_bias"}
                }
            }
        }
    }

    # Baseline CNN file
    cnn_file = find_file("cnn_predictions_era5_to_mswep.nc")

    # ----------------------------------------------------
    # UNET CONFIG
    # ----------------------------------------------------
    unet_cfg = eval(repr(base_config))
    unet_cfg["experiment"] = "eval_unet_experiments"
    
    if cnn_file:
        unet_cfg["datasets"].append({"name": "CNN_Base", "file_path": cnn_file, "variable_name": "precipitation"})
    
    # Retained U-Net
    unet_retained = find_file("unet_coordconv_predictions_era5_to_mswep.nc")
    if unet_retained:
        unet_cfg["datasets"].append({"name": "UNet_Retained", "file_path": unet_retained, "variable_name": "precipitation"})
        unet_cfg["visualisation"]["model_colors"]["UNet_Retained"] = "#1D3557"
        
    for i in range(1, 11):
        f = find_file(f"unet_exp{i}_predictions_era5_to_mswep.nc")
        if f:
            name = f"UNet_Exp{i}"
            unet_cfg["datasets"].append({"name": name, "file_path": f, "variable_name": "precipitation"})
            # Assign shades of blue/purple for variations
            unet_cfg["visualisation"]["model_colors"][name] = f"#{30+i*2}{50+i*3}{100+i*10}"

    unet_config_path = "../../configs/experiments/config_unet_experiments.yaml"
    with open(unet_config_path, "w") as f:
        yaml.dump(unet_cfg, f, default_flow_style=False, sort_keys=False)

    # ----------------------------------------------------
    # VIT CONFIG
    # ----------------------------------------------------
    vit_cfg = eval(repr(base_config))
    vit_cfg["experiment"] = "eval_vit_experiments"
    
    if cnn_file:
        vit_cfg["datasets"].append({"name": "CNN_Base", "file_path": cnn_file, "variable_name": "precipitation"})
    
    # Retained ViT
    vit_retained = find_file("vit_predictions_era5_to_mswep.nc")
    if vit_retained:
        vit_cfg["datasets"].append({"name": "ViT_Retained", "file_path": vit_retained, "variable_name": "precipitation"})
        vit_cfg["visualisation"]["model_colors"]["ViT_Retained"] = "#E63946"
        
    for i in range(1, 9):
        f = find_file(f"vit_exp{i}_predictions_era5_to_mswep.nc")
        if f:
            name = f"ViT_Exp{i}"
            vit_cfg["datasets"].append({"name": name, "file_path": f, "variable_name": "precipitation"})
            # Assign shades of red/orange for variations
            vit_cfg["visualisation"]["model_colors"][name] = f"#{200-i*10}{50+i*5}{50+i*5}"

    vit_config_path = "../../configs/experiments/config_vit_experiments.yaml"
    with open(vit_config_path, "w") as f:
        yaml.dump(vit_cfg, f, default_flow_style=False, sort_keys=False)

    print(f"Generated {unet_config_path}")
    print(f"Generated {vit_config_path}")

    # ----------------------------------------------------
    # SLURM SCRIPT
    # ----------------------------------------------------
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=eval_postproc
#SBATCH --output=logs/eval_postproc_%j.log
#SBATCH --error=logs/eval_postproc_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1
cd ../..

echo "======================================"
echo "Evaluating UNet Experiments"
echo "======================================"
python3 -u src/postproc.py configs/experiments/config_unet_experiments.yaml

echo "======================================"
echo "Evaluating ViT Experiments"
echo "======================================"
python3 -u src/postproc.py configs/experiments/config_vit_experiments.yaml

echo "All Postprocessing Completed!"
