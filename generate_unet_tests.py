# Author: M. El Aabaribaoune (@um6p)
import os
import yaml

base_yaml_str = """
general:
  verbose: true
  experiment: "unet_exp1"
  variable: "precip"  
  target: "mswep"    
  src: "era5"        
  model_type: "unet_v2" 
  interpolation_type: "linear" 
  variables: ["z", "q", "t", "u", "v"]
  levels: [500, 700, 850, 1000]
  resolution: 2.0

training:
  learning_rate: 1e-3
  epochs: 150
  batch_size: 64
  loss_type: "bernoulli_gamma" 
  norm_mode: "gridbox"         
  early_stopping:
    enable: true
    max: 20
  LR_scheduler:
    enable: true
    patience: 10
    factor: 0.5
    min_lr: 1e-6
  gradient_clipping:
    enable: true
    value: 1.0
  weight_decay:
    enable: true
    value: 1e-4
  optimizer: "adamw"
  dropout:
    enable: true
    value: 0.1
  scheduler: 
    enable: false
    type: "cosine"
  group_norm:
    enable: true
    num_groups: 32
  validation:
    enable: true
    pecentage: 0.2

region:
  lon_min: -18.0
  lon_max: 0.0
  lat_min: 21.0
  lat_max: 37.0

dates:
  train:
    start: "1979-01-01"
    end: "1983-12-31"
  test:
    start: "1984-01-01"
    end: "1984-12-31"

plots:
  eval:
    show_suffix_components_in_title: true

paths:
  root_dir: "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/era5ztquv/1979_2020/all_data"
  results_dir: "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/"
  shapefile_path: "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/shapefiles/Morocco_shpfile/DA_REGIONS_12R.shp"
  era5_predictor_pattern: "{var}_1979-2020_levels.nc"
  mswep_path: "mswep_1979_2020.nc"
"""

experiments = {
    "test_exp1": {"experiment": "unet_exp1_v2_bg", "model_type": "unet_v2", "loss_type": "bernoulli_gamma"},
    "test_exp2": {"experiment": "unet_exp2_coordconv_bg", "model_type": "unet_coordconv", "loss_type": "bernoulli_gamma"},
    "test_exp3": {"experiment": "unet_exp3_attention_bg", "model_type": "attention_unet", "loss_type": "bernoulli_gamma"},
    "test_exp4": {"experiment": "unet_exp4_coordconv_mse", "model_type": "unet_coordconv", "loss_type": "mse", "learning_rate": 5e-4, "batch_size": 128},
    "test_exp5": {"experiment": "unet_exp5_coordconv_cosine", "model_type": "unet_coordconv", "loss_type": "bernoulli_gamma", "scheduler_enable": True},
    
    # Doury Emulator tests
    "test_exp6": {"experiment": "unet_exp6_doury_asymmetric", "model_type": "doury_unet", "loss_type": "asymmetric_mse", "learning_rate": 5e-4},
    "test_exp7": {"experiment": "unet_exp7_doury_weighted", "model_type": "doury_unet", "loss_type": "intensity_weighted_mse", "learning_rate": 5e-4},
    "test_exp8": {"experiment": "unet_exp8_doury_hurdle", "model_type": "doury_unet", "loss_type": "hurdle_loss", "learning_rate": 1e-3},
    "test_exp9": {"experiment": "unet_exp9_doury_bg", "model_type": "doury_unet", "loss_type": "bernoulli_gamma", "learning_rate": 1e-3},
    
    # LR Grid Search on Doury_UNet + HurdleLoss
    "test_exp10": {"experiment": "unet_exp10_lr_1e-2", "model_type": "doury_unet", "loss_type": "hurdle_loss", "learning_rate": 1e-2},
    "test_exp11": {"experiment": "unet_exp11_lr_5e-4", "model_type": "doury_unet", "loss_type": "hurdle_loss", "learning_rate": 5e-4},
    "test_exp12": {"experiment": "unet_exp12_lr_1e-4", "model_type": "doury_unet", "loss_type": "hurdle_loss", "learning_rate": 1e-4},
}

os.makedirs("main/configs/unet/tests", exist_ok=True)
base_dict = yaml.safe_load(base_yaml_str)

for exp_name, mods in experiments.items():
    d = yaml.safe_load(base_yaml_str)
    d["general"]["experiment"] = mods["experiment"]
    d["general"]["model_type"] = mods["model_type"]
    d["training"]["loss_type"] = mods["loss_type"]
    
    if "learning_rate" in mods:
        d["training"]["learning_rate"] = mods["learning_rate"]
    if "batch_size" in mods:
        d["training"]["batch_size"] = mods["batch_size"]
    if "scheduler_enable" in mods:
        d["training"]["scheduler"]["enable"] = mods["scheduler_enable"]
        
    with open(f"main/configs/unet/tests/{exp_name}.yaml", "w") as f:
        yaml.dump(d, f, sort_keys=False)

print("Generated 5 configurations.")
