import xarray as xr
import sys
try:
    ds = xr.open_dataset("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/Intern_Hamza/elevation.tif", engine="rasterio")
    print(ds)
except Exception as e:
    print("Error:", e)
