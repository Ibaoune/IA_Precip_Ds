import yaml
import glob
import os

configs = glob.glob("main/configs/*/retained/config.yaml")

for cfg_path in configs:
    with open(cfg_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Fix paths
    if 'paths' in data:
        for k, v in data['paths'].items():
            if isinstance(v, str) and '/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/' in v:
                data['paths'][k] = v.replace('/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/', '/srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/')
        
        # specifically fix the duplicated mswep_path
        if 'mswep_path' in data['paths']:
            bad_part = '/srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/reanalysis/era5//home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/obs/mswep/mswep_1979_2020.nc'
            if bad_part in data['paths']['mswep_path'] or data['paths']['mswep_path'] == bad_part:
                data['paths']['mswep_path'] = '/srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/obs/mswep/mswep_1979_2020.nc'
                
            # If it has the new duplicate:
            dup_part = '/srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/reanalysis/era5//srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/obs/mswep/mswep_1979_2020.nc'
            if data['paths']['mswep_path'] == dup_part:
                data['paths']['mswep_path'] = '/srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/obs/mswep/mswep_1979_2020.nc'
                
            # Fallback direct fix:
            if 'mswep' in data['paths']['mswep_path']:
                data['paths']['mswep_path'] = '/srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/obs/mswep/mswep_1979_2020.nc'

    with open(cfg_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Fixed {cfg_path}")
