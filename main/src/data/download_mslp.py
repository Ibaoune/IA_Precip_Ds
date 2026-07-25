import cdsapi
import os

def download_mslp():
    c = cdsapi.Client()
    
    out_dir = '/srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/reanalysis/era5'
    out_file = os.path.join(out_dir, 'msl_1979-2020_levels.nc')
    
    if os.path.exists(out_file):
        print(f"File {out_file} already exists.")
        return

    print("Downloading MSLP for 1979-2020 (12:00) over domain (N:37, W:-18, S:21, E:0)...")
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': 'mean_sea_level_pressure',
            'year': [str(y) for y in range(1979, 2021)],
            'month': [str(m).zfill(2) for m in range(1, 13)],
            'day': [str(d).zfill(2) for d in range(1, 32)],
            'time': '12:00',
            'area': [37, -18, 21, 0],
            'format': 'netcdf',
        },
        out_file
    )
    print("Download completed successfully.")

if __name__ == '__main__':
    download_mslp()
