import yaml
import os

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update general settings
config['general']['model_type'] = 'cnn'
config['general']['experiment'] = 'cnn_test_inference'

# Update paths
config['paths']['lmdz_predictor_pattern'] = "{folder}/{lmdz_var}-hist.nc"
config['paths']['root_dir'] = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/era5ztquv/1979_2020/all_data"
config['paths']['shapefile_path'] = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/shapefiles/Morocco_shpfile/DA_REGIONS_12R.shp"
config['paths']['results_dir'] = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/"
config['paths']['model_path'] = "/srv/data/mohammad.elaabaribao/work/interns/y2026/code/era5Tomswep/results/cnn_exp2/region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_5e-05_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn/models/cnn_precip.pth"

# Update prediction scenarios
config['prediction']['models_dir'] = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/"

scenarios = config['prediction']['scenarios']
for scenario in scenarios:
    if scenario['name'] == 'era5_present':
        scenario['folder'] = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/era5ztquv/1979_2020/all_data"
    elif scenario['name'] == 'lmdz_35_present':
        scenario['folder'] = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/LMDZ/r35"
        scenario['bc_reference_folder'] = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/LMDZ/r35"
    elif scenario['name'] == 'lmdz_250_present':
        scenario['folder'] = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/LMDZ/r250"
        scenario['bc_reference_folder'] = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/LMDZ/r250"

with open('config.yaml', 'w') as f:
    yaml.dump(config, f, sort_keys=False)
