# Author: M. El Aabaribaoune (@um6p)
import yaml
import os

template_file = "config_vit_exp11_1979_2014.yaml"
with open(template_file, "r") as f:
 template = yaml.safe_load(f)

models = [
 {
 "name": "vit_precip_exp21_best_hybrid",
 "path": "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/tests/vit/vit_precip_exp21_best_hybrid/region_lat_21.0_36.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/global_0.001_bernoulli_gamma_25ep_lr_sched_wd_gc_dropout_cosine_gn"
 },
 {
 "name": "vit_precip_exp21_hybrid_base",
 "path": "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/tests/vit/vit_precip_exp21_hybrid_base/region_lat_21.0_36.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/global_0.001_bernoulli_gamma_25ep_lr_sched_wd_gc_dropout_cosine_gn"
 },
 {
 "name": "vit_precip_exp22_hybrid_deep_reg",
 "path": "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/tests/vit/vit_precip_exp22_hybrid_deep_reg/region_lat_21.0_36.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/global_0.0005_bernoulli_gamma_30ep_lr_sched_wd_gc_dropout_cosine_gn"
 },
 {
 "name": "vit_precip_exp23_hybrid_bilinear_channel",
 "path": "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/tests/vit/vit_precip_exp23_hybrid_bilinear_channel/region_lat_21.0_36.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/channel_0.001_bernoulli_gamma_25ep_lr_sched_wd_gc_dropout_cosine_gn"
 }
]

for model in models:
 cfg = yaml.safe_load(yaml.dump(template)) # deep copy
 name = model["name"]
 base_path = model["path"]
 
 cfg["general"]["experiment"] = name
 cfg["paths"]["model_path"] = os.path.join(base_path, "models", "vit_precip.pth")
 cfg["prediction"]["train_config_path"] = os.path.join(base_path, "config.txt")
 cfg["prediction"]["output_dir"] = f"results/1979_2014/{name}"
 
 # Enable ONLY lmdz_250_present
 for scenario in cfg["prediction"]["scenarios"]:
 if scenario["name"] == "lmdz_250_present":
 scenario["enable"] = True
 else:
 scenario["enable"] = False
 
 out_file = f"config_{name}_1979_2014.yaml"
 with open(out_file, "w") as f:
 yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
 print(f"Created {out_file}")
