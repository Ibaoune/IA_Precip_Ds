"""
Author: M. El Aabaribaoune (@um6p)
Description: Utility functions for data processing, statistical analysis, and plotting.
"""


import os
import shutil
import glob
import yaml

configs_src = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/configs/experiments"
results_src = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/experiments"

vit_configs_dst = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/configs/vit/tests/tests_early_experiments"
vit_results_dst = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/vit/tests/tests_early_experiments"
unet_configs_dst = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/configs/unet/tests/tests_early_experiments"
unet_results_dst = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/unet/tests/tests_early_experiments"

os.makedirs(vit_configs_dst, exist_ok=True)
os.makedirs(vit_results_dst, exist_ok=True)
os.makedirs(unet_configs_dst, exist_ok=True)
os.makedirs(unet_results_dst, exist_ok=True)

# Write readmes
with open(os.path.join(vit_configs_dst, "README.md"), "w") as f:
    f.write("# Early ViT Experiments\n\nThese are the initial vit_exp1 through vit_exp8 legacy configurations.")
with open(os.path.join(unet_configs_dst, "README.md"), "w") as f:
    f.write("# Early UNet Experiments\n\nInitial U-Net tests (e.g. test_cpu).")

# Move configs
for i in range(1, 9):
    cfg_name = f"vit_exp{i}.yaml"
    src = os.path.join(configs_src, cfg_name)
    if os.path.exists(src):
        shutil.move(src, os.path.join(vit_configs_dst, cfg_name))

if os.path.exists(os.path.join(configs_src, "test_cpu.yaml")):
    shutil.move(os.path.join(configs_src, "test_cpu.yaml"), os.path.join(unet_configs_dst, "test_cpu.yaml"))

# Move results and check if they have .nc files
def move_result(exp_name, dst_dir, model_name):
    src = os.path.join(results_src, exp_name)
    yaml_not_found = f"/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/{model_name}/tests/yaml_not_found"
    os.makedirs(yaml_not_found, exist_ok=True)
    
    if os.path.exists(src):
        nc_files = glob.glob(f"{src}/**/*.nc", recursive=True)
        if len(nc_files) > 0:
            shutil.move(src, os.path.join(dst_dir, exp_name))
        else:
            shutil.move(src, os.path.join(yaml_not_found, exp_name))

for i in range(1, 9):
    move_result(f"vit_exp{i}", vit_results_dst, "vit")

move_result("test_cpu", unet_results_dst, "unet")

print("Early experiments successfully moved!")
