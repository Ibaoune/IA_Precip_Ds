"""
Author: M. El Aabaribaoune (@um6p)
Description: Part of the downscaling inference engine.
"""

import yaml
import os

base_yaml = "config_cnn.yaml"
with open(base_yaml, "r") as f:
    cfg = yaml.safe_load(f)

tasks = [
    {
        "name": "cnn_exp3",
        "train_config_path": "../../../interns/y2026/code/era5Tomswep/results/tests/cnn/cnn_exp3/region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn/config.txt",
        "output_dir": "results/output/cnn_exp3"
    },
    {
        "name": "cnn_exp5",
        "train_config_path": "../../../interns/y2026/code/era5Tomswep/results/tests/cnn/cnn_exp5/region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.0001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn/config.txt",
        "output_dir": "results/output/cnn_exp5"
    },
    {
        "name": "vit_exp2",
        "train_config_path": "../../../interns/y2026/code/era5Tomswep/results/vit_exp2/region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.0005_bernoulli_gamma_100ep_wd_gc_dropout/config.txt",
        "output_dir": "results/output/vit_exp2"
    },
    {
        "name": "glm_precip_l2",
        "train_config_path": "../main/results/tests/glm/glm_precip_l2/region_lat_21.0_36.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.001_bernoulli_gamma_10ep_wd/config.txt",
        "output_dir": "results/output/glm_precip_l2"
    }
]

for t in tasks:
    new_cfg = dict(cfg) # shallow copy
    # deep copy the dicts we modify
    new_cfg["prediction"] = dict(cfg["prediction"])
    new_cfg["general"] = dict(cfg["general"])
    
    new_cfg["prediction"]["train_config_path"] = t["train_config_path"]
    new_cfg["prediction"]["output_dir"] = t["output_dir"]
    
    # change general model_type to match
    if "vit" in t["name"]:
        new_cfg["general"]["model_type"] = "vit"
    elif "glm" in t["name"]:
        new_cfg["general"]["model_type"] = "glm"
    else:
        new_cfg["general"]["model_type"] = "cnn"

    yaml_name = f"config_{t['name']}.yaml"
    with open(yaml_name, "w") as f:
        yaml.dump(new_cfg, f, sort_keys=False)
