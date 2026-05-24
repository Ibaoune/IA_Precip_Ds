# Author: M. El Aabaribaoune (@um6p)
import xarray as xr

# Load predictions
ds_mod = xr.open_dataset('../inference/results/output/raw_lmdz35_10km.nc')
var_mod = ds_mod["precipitation"]

# Load reference
ds_ref = xr.open_dataset('/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/era5ztquv/1979_2020/all_data/mswep_1979_2020.nc')
var_ref = ds_ref["precipitation"].sel(time=slice('2006-01-01', '2020-12-31'))

var_mod['time'] = var_mod['time'].dt.floor('D')
var_ref['time'] = var_ref['time'].dt.floor('D')

print("Mod time start/end:", var_mod.time.values[0], var_mod.time.values[-1])
print("Ref time start/end:", var_ref.time.values[0], var_ref.time.values[-1])

var_mod_aligned, var_ref_aligned = xr.align(var_mod, var_ref, join='inner')

print("Aligned mod time size:", var_mod_aligned.time.size)
print("Aligned ref time size:", var_ref_aligned.time.size)

