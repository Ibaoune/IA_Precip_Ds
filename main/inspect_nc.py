import xarray as xr
import sys

files = {
    "CNN": "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/cnn/retained/cnn_exp5/region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.0001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn/output_data/cnn_predictions_era5_to_mswep.nc",
    "GLM": "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/glm/retained/glm_precip_l2/region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.001_bernoulli_gamma_10ep_wd/output_data/glm_predictions_era5_to_mswep.nc",
    "U-Net": "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/unet/retained/unet_exp4_coordconv_mse/region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.001_mse_150ep_lr_sched_wd_gc_dropout_gn/output_data/unet_coordconv_predictions_era5_to_mswep.nc",
    "ViT": "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/vit/retained/vit_precip_exp21_best_hybrid/region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/global_0.001_bernoulli_gamma_25ep_lr_sched_wd_gc_dropout_cosine_gn/output_data/vit_predictions_era5_to_mswep.nc"
}

for name, path in files.items():
    print(f"=== {name} ===")
    try:
        ds = xr.open_dataset(path)
        print(f"Exists: Yes")
        print(f"Dimensions: {list(ds.dims.keys())}")
        print(f"Coordinate names: {list(ds.coords.keys())}")
        print(f"Available variables: {list(ds.data_vars.keys())}")
        
        # prediction variable name
        var_name = None
        for v in ["precipitation", "pr", "tp", "prediction", "pred"]:
            if v in ds.data_vars:
                var_name = v
                break
        if var_name is None:
            var_name = list(ds.data_vars.keys())[0] if len(ds.data_vars) > 0 else "None"
            
        print(f"Prediction variable name: {var_name}")
        if var_name in ds.data_vars:
            print(f"Units: {ds[var_name].attrs.get('units', 'No units attribute')}")
            
            # missing value structure
            nans = ds[var_name].isnull().sum().item()
            print(f"Missing-value structure: {nans} NaNs found in variable")
            
        if "time" in ds.coords:
            time_vals = ds.time.values
            print(f"Time start: {time_vals[0]}")
            print(f"Time end: {time_vals[-1]}")
            
        if "lat" in ds.coords:
            lat_vals = ds.lat.values
            print(f"Latitude range: [{lat_vals.min()}, {lat_vals.max()}]")
        if "lon" in ds.coords:
            lon_vals = ds.lon.values
            print(f"Longitude range: [{lon_vals.min()}, {lon_vals.max()}]")
            
    except Exception as e:
        print(f"Exists: No (or error opening: {e})")
    print()
