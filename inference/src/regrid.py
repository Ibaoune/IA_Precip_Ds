"""
Author: M. El Aabaribaoune (@um6p)
Description: Part of the downscaling inference engine.
"""

import xarray as xr
import os

print("Loading MSWEP (target grid)...")
mswep_path = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/era5ztquv/1979_2020/all_data/mswep_1979_2020.nc"
mswep = xr.open_dataset(mswep_path)

print("Loading LMDZ35 precipitation...")
lmdz_path = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/LMDZ/r35/precip-hist.nc"
lmdz = xr.open_dataset(lmdz_path)

print("Converting units and renaming variable...")
# Convert from kg/m2/s to mm/day by multiplying by 86400
precip = lmdz['precip'] * 86400.0
precip.attrs = lmdz['precip'].attrs
precip.attrs['units'] = 'mm/day'

# Rename variable to 'precipitation' to match CNN outputs
ds_lmdz = xr.Dataset({'precipitation': precip})

print("Regridding LMDZ35 to MSWEP grid...")
# Interpolate LMDZ35 to MSWEP's spatial grid. 
# It's important to rename coordinates if they differ, but here they are 'lat' and 'lon'.
ds_regridded = ds_lmdz.interp(lat=mswep.lat, lon=mswep.lon, method='linear')

# Optionally align time coordinates names if needed
if 'time_counter' in ds_regridded.coords:
    ds_regridded = ds_regridded.rename({'time_counter': 'time'})

out_path = "results/output/raw_lmdz35_10km.nc"
print(f"Saving to {out_path}...")
ds_regridded.to_netcdf(out_path)
print("Done!")
