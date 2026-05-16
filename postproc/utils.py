import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cartopy.crs as ccrs
import matplotlib as mpl

# Constants
DEFAULT_SHAPEFILE = os.path.join(os.path.dirname(__file__), "shape file/morocco_unified_fixed_v2.shp")
PROJECT_ROOT = os.path.dirname(__file__)
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def get_current_region():
    return os.environ.get("POSTPROC_REGION", "allmorr")

def get_shapefile(region=None):
    if region is None:
        region = get_current_region()
    mapping = {
        "allmorr": os.path.join(os.path.dirname(__file__), "shape file/morocco_unified_fixed_v2.shp"),
        "north": os.path.join(os.path.dirname(__file__), "shape file/north.shp"),
        "north_east": os.path.join(os.path.dirname(__file__), "shape file/north_east.shp"),
        "south": os.path.join(os.path.dirname(__file__), "shape file/south.shp"),
        "east": os.path.join(os.path.dirname(__file__), "shape file/east.shp"),
    }
    return mapping.get(region, mapping["allmorr"])

def load_config(config_path):
    import yaml
    # Load the requested config
    with open(config_path, 'r') as f:
        local_cfg = yaml.safe_load(f)
    
    # If it's a local metric config (contains 'metric' but missing shared fields like 'experiment')
    # then merge it with the global one.
    if local_cfg and 'metric' in local_cfg and 'experiment' not in local_cfg:
        global_path = os.environ.get("POSTPROC_MASTER_CONFIG", os.path.join(PROJECT_ROOT, "config.yaml"))
        if os.path.exists(global_path):
            with open(global_path, 'r') as f:
                global_cfg = yaml.safe_load(f)
            # Merge: global provides the base, local provides the 'metric'
            global_cfg['metric'] = local_cfg['metric']
            local_cfg = global_cfg
            
    # Inject current region from environment if set
    if os.environ.get("POSTPROC_REGION") and local_cfg and 'parameters' in local_cfg:
        local_cfg['parameters']['region'] = os.environ.get("POSTPROC_REGION")
        
    return local_cfg

def get_title_metadata(metric_name, period=""):
    config_path = os.environ.get("POSTPROC_MASTER_CONFIG", os.path.join(PROJECT_ROOT, "config.yaml"))
    show_metadata = False
    start_date = "Unknown"
    end_date = "Unknown"
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            global_cfg = yaml.safe_load(f)
            if global_cfg and 'parameters' in global_cfg:
                show_metadata = global_cfg['parameters'].get('show_title_metadata', False)
                start_date = global_cfg['parameters'].get('start_date', start_date)
                end_date = global_cfg['parameters'].get('end_date', end_date)
                
    if not show_metadata:
        return ""
        
    METRIC_METADATA = {
        "mean": {"what": "Average daily precipitation", "how": "Mean of daily precipitation over the specified period"},
        "bias": {"what": "Mean Error (Bias)", "how": "Model precipitation minus Reference precipitation"},
        "rmse": {"what": "Root Mean Square Error", "how": "Square root of the average of squared differences"},
        "correlation": {"what": "Pearson Correlation", "how": "Linear correlation between Model and Reference"},
        "cdd": {"what": "Consecutive Dry Days", "how": "Max consecutive days with precipitation < 1mm/day"},
        "qqplot": {"what": "Q-Q Plot", "how": "Quantile-Quantile distribution comparison"},
        "r01": {"what": "Wet Days (PR >= 1mm)", "how": "Count of days with precipitation >= 1mm/day"},
        "r95_freq": {"what": "Heavy Precip Frequency", "how": "Number of days > 95th percentile"},
        "r99_freq": {"what": "Extreme Precip Frequency", "how": "Number of days > 99th percentile"},
        "r95p": {"what": "Heavy Precip Frequency", "how": "Number of days > 95th percentile"},
        "r99p": {"what": "Extreme Precip Frequency", "how": "Number of days > 99th percentile"},
        "r95": {"what": "Heavy Precipitation", "how": "Precipitation amount on days > 95th percentile"},
        "r99": {"what": "Extreme Precipitation", "how": "Precipitation amount on days > 99th percentile"},
        "nbevents": {"what": "Frequency of events", "how": "Number of extreme precipitation events"},
        "rocss": {"what": "ROCSS", "how": "Relative Operating Characteristic Skill Score"},
        "intensity": {"what": "Daily Intensity Distribution", "how": "Probability Density Function of daily intensities"}
    }
    
    base_name = str(metric_name).lower()
    match = None
    for k in sorted(METRIC_METADATA.keys(), key=len, reverse=True):
        if k in base_name:
            match = k
            break
            
    period_str = f"{start_date} to {end_date}"
    if period and period not in ["Annual Cycle", "All Days", "all"]:
        period_str += f" ({period})"
    elif period:
        period_str += f" ({period})"
        
    if match:
        what = METRIC_METADATA[match]["what"]
        how = METRIC_METADATA[match]["how"]
        return f"\n[ Period: {period_str} | {what} | {how} ]"
    
    return f"\n[ Period: {period_str} ]"

def use_robust_limits():
    config_path = os.environ.get("POSTPROC_MASTER_CONFIG", os.path.join(PROJECT_ROOT, "config.yaml"))
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            global_cfg = yaml.safe_load(f)
            if global_cfg and 'parameters' in global_cfg:
                return global_cfg['parameters'].get('impose_robust_limits', True)
    return True

def get_custom_limits(metric_name, plot_type):
    config_path = os.environ.get("POSTPROC_MASTER_CONFIG", os.path.join(PROJECT_ROOT, "config.yaml"))
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            global_cfg = yaml.safe_load(f)
            if global_cfg and 'parameters' in global_cfg:
                if not global_cfg['parameters'].get('impose_robust_limits', True):
                    return None
            custom = global_cfg.get('custom_limits', {})
            base_name = str(metric_name).lower()
            match = None
            for k in sorted(custom.keys(), key=len, reverse=True):
                if k in base_name:
                    match = k
                    break
            if match and plot_type in custom[match]:
                limits = custom[match][plot_type]
                if isinstance(limits, list) and len(limits) >= 2:
                    return limits
    return None

def get_results_dir(config, metric_name, root_path):
    """
    Construct results path based on the requested structure.
    """
    exp_name = config.get('experiment', 'default_exp')
    model_name = config.get('model')
    period = config.get('period', 'present')
    region = config.get('parameters', {}).get('region', 'allmorr')
    
    start_date = config.get('parameters', {}).get('start_date', 'unknown')
    end_date = config.get('parameters', {}).get('end_date', 'unknown')
    date_range = f"{start_date}_{end_date}"
    
    # Exploration Case
    if "exploration" in exp_name or metric_name == "explore":
        return os.path.join(root_path, "results", exp_name, date_range, region, "explore")
    
    # Standard Case (ERA5/UNET)
    return os.path.join(root_path, "results", exp_name, date_range, region, "test", metric_name)

# Plot Styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Color Palettes
BIAS_PURPLE_ORANGE = [
    '#1a0033', '#2d004b', '#542788', '#8073ac', '#b2abd2', '#d8daeb', 
    '#ffffff', '#ffffff', # Two white slots to cover the larger 0 bin
    '#fee0b6', '#fdb863', '#e08214', '#b35806', '#7f3b08', '#4d2204'
]
BIAS_RDBU_WHITE = [
    '#053061', '#1a5899', '#337eb8', '#5ca3cb', '#96c7df', '#c7e0ed', 
    '#ffffff', '#ffffff', 
    '#fcd3bc', '#f5a886', '#df765e', '#c53e3d', '#9f1228', '#67001f'
]
BIAS_LEVELS = [-2, -1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5, 2]

# For Extreme Bias (Errors in R95, R99, CDD)
BIAS_EXTREME_LEVELS = [-50, -30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 30, 50]

GLOBAL_MODEL_COLORS = {
    'MSWEP': '#000000', # Black
    'CNN_EXP3': '#A8DADC', # Very Light Blue
    'CNN_EXP5': '#457B9D', # Light Blue
    'VIT_EXP21_BEST': '#E63946', # Red
    'VIT_PRECIP_EXP22_HYBRID_DEEP_REG': '#9B2226', # Dark Red
    'GLM_LMDZ250': '#2A9D8F', # Green
    'GLM_ERA5': '#264653', # Dark Teal
    'VIT': '#E63946',
    'CNN': '#457B9D',
    'UNET': '#1D3557',
    'GLM': '#2A9D8F',
}

VIRIDIS_PALETTE = [
    '#440154', '#482878', '#3e4989', '#31688e', '#26828e',
    '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'
]

# Specifically for Precipitation Mean (First bin < 0.25 is white)
PRECIP_MEAN_PALETTE = [
    '#ffffff', # White for < 0.25
    '#440154', '#482878', '#3e4989', '#31688e', '#21908d', 
    '#35b779', '#8fd744', '#fde725', '#fb9a99', '#e31a1c', '#800000' # Added dark red for extreme max
]
PRECIP_MEAN_LEVELS = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]

# For Extreme Intensity (r95, r99) - Units: mm/day
# Extended range to capture heavy events
PRECIP_EXTREME_LEVELS = [0, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]

# For Extreme Frequency (r95p, r99p, nbEvents) - Units: Days
# Represents number of events per period
FREQ_LEVELS = [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60]

# For Dry Periods (CDD) - Units: Consecutive Days
CDD_LEVELS = [0, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300]

def get_custom_cmap(name):
    if name == "bias" or name == "correlation":
        return ListedColormap(BIAS_PURPLE_ORANGE), BIAS_PURPLE_ORANGE
    else:
        return ListedColormap(VIRIDIS_PALETTE), VIRIDIS_PALETTE

SEASONS = {
    'Annual': list(range(1, 13)),
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11]
}

REGION_BOUNDS = {
    "north": {"lon": [-10, 0], "lat": [29, 36]},
    "south": {"lon": [-10, 0], "lat": [21, 29]},
    "allmorr": {"lon": [-18, 0], "lat": [21, 37]},
}

def mask_dataset(ds, region=None, only_land=False, only_morocco=True, shapefile=None):
    """
    Filters and masks a dataset based on geographic regions and optional land/country masks.
    """
    if region is None:
        region = get_current_region()
    if shapefile is None or shapefile == DEFAULT_SHAPEFILE:
        shapefile = get_shapefile(region)
        
    bounds = REGION_BOUNDS.get(region, REGION_BOUNDS["allmorr"])
    lon_range = slice(bounds["lon"][0], bounds["lon"][1])
    lat_range = slice(bounds["lat"][0], bounds["lat"][1])

    var_lon = next(v for v in ds.variables if v.startswith("lon"))
    var_lat = next(v for v in ds.variables if v.startswith("lat"))
    geo_mask = (
        (ds[var_lon] >= lon_range.start) & (ds[var_lon] <= lon_range.stop) &
        (ds[var_lat] >= lat_range.start) & (ds[var_lat] <= lat_range.stop)
    )
    ds = ds.where(geo_mask, drop=True)
    
    if only_land:
        land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
        land_mask = land.mask(ds)
        ds = ds.where(~land_mask.isnull(), drop=True)
        
    if only_morocco:
        morocco = gpd.read_file(shapefile).to_crs("EPSG:4326").dissolve()
        morocco["name"] = ["morocco"]
        morocco = morocco.reset_index(drop=True)
        morocco_mask = regionmask.from_geopandas(morocco, names="name", name="morocco")
        mask = morocco_mask.mask(ds)
        
        # Explicitly set values outside to NaN and drop if necessary (or just set where to mask)
        ds = ds.where(~mask.isnull())
    return ds

def get_time_info(time_array):
    """
    Extracts numerical year and month information from a datetime array.
    """
    ts = pd.to_datetime(time_array)
    return ts.year.values, ts.month.values

def get_global_limits(local_vmin, local_vmax, current_region, save_path, period="Annual"):
    """
    Infers the true global limits by loading the 'allmorr' NetCDF files.
    """
    if current_region == "allmorr" or save_path is None:
        return local_vmin, local_vmax
        
    try:
        allmorr_dir = save_path.replace(f"/{current_region}/", "/allmorr/")
        allmorr_dir = os.path.dirname(os.path.dirname(allmorr_dir)) # Go up from figures/period/ to metric dir
        
        import glob
        nc_files = glob.glob(os.path.join(allmorr_dir, f"*_{period}.nc"))
        if not nc_files:
            return local_vmin, local_vmax
            
        g_min, g_max = local_vmin, local_vmax
        for f in nc_files:
            ds = xr.open_dataset(f)
            var = list(ds.data_vars.values())[0]
            vals = var.values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                if use_robust_limits():
                    g_max = max(g_max, np.percentile(vals, 98))
                    g_min = min(g_min, np.percentile(vals, 2))
                else:
                    g_max = max(g_max, vals.max())
                    g_min = min(g_min, vals.min())
        return g_min, g_max
    except Exception as e:
        return local_vmin, local_vmax

def get_global_evolution_limits(current_region, save_path, period="Annual"):
    """
    Infers the true global y-axis limits for temporal evolution by loading 'allmorr' NetCDF files.
    """
    if current_region == "allmorr" or save_path is None:
        return None, None
    try:
        allmorr_dir = save_path.replace(f"/{current_region}/", "/allmorr/")
        allmorr_dir = os.path.dirname(os.path.dirname(allmorr_dir))
        
        import glob
        nc_files = glob.glob(os.path.join(allmorr_dir, f"*_{period}.nc"))
        if not nc_files:
            return None, None
            
        g_min, g_max = np.inf, -np.inf
        allmorr_shape = get_shapefile("allmorr")
        for f in nc_files:
            ds = xr.open_dataset(f)
            var = list(ds.data_vars.values())[0]
            means, _ = get_regional_means(var, allmorr_shape)
            if len(means) > 0:
                if use_robust_limits():
                    g_max = max(g_max, np.nanpercentile(means, 98))
                    g_min = min(g_min, np.nanpercentile(means, 2))
                else:
                    g_max = max(g_max, np.nanmax(means))
                    g_min = min(g_min, np.nanmin(means))
        if g_min == np.inf:
            return None, None
        
        # Add 5% padding
        padding = (g_max - g_min) * 0.05
        return g_min - padding, g_max + padding
    except Exception as e:
        return None, None

def save_metrics_to_netcdf(metric_dict, filename_prefix, output_dir, tag):
    """
    Consolidates a dictionary of metric arrays into a single NetCDF file.
    
    Inputs:
        metric_dict (dict): Dictionary where keys are metric names and values are xr.DataArrays.
        filename_prefix (str): Base name for the file.
        output_dir (str): Directory where the file will be saved.
        tag (str): Suffix for the filename (e.g., 'Annual', 'DJF').
        
    Outputs:
        None (Saves file to disk).
        
    How it works:
        1. Initializes an empty xarray Dataset.
        2. Iterates through the dictionary, adding each DataArray as a variable.
        3. Exports the dataset to a .nc file in the specified directory.
    """
    dataset = xr.Dataset()
    for name, da in metric_dict.items():
        dataset[name.lower()] = da
    path = os.path.join(output_dir, f"{filename_prefix}_{tag}.nc")
    dataset.to_netcdf(path)
    print(f"Saved metrics to: {path}")


def plot_spatial_maps(data_dict, metric_name, period="Annual", shapefile=None, save_path=None, title=None, levels=None, unit="mm/day", nrows=1, region=None):
    """
    Creates a high-quality spatial comparison plot for multiple models following scientific visualization rules.
    """
    if region is None:
        region = get_current_region()
    if shapefile is None or shapefile == DEFAULT_SHAPEFILE:
        shapefile = get_shapefile(region)
        
    morocco_gdf = gpd.read_file(shapefile).dissolve()
    morocco = morocco_gdf.boundary
    bounds = morocco_gdf.total_bounds
    cmap, _ = get_custom_cmap(metric_name)
    
    n_models = len(data_dict)
    ncols = int(np.ceil(n_models / nrows))
    fig = plt.figure(figsize=(5 * ncols, 7 * nrows), dpi=300)
    
    # Title hierarchy (Rule 4.1)
    fig_title = title if title else f"{metric_name.upper()} Spatial Distribution ({period})"
    fig_title += get_title_metadata(metric_name, period)
    plt.suptitle(fig_title, fontsize=12, fontweight='bold', y=0.98)

    # Calculate local range
    all_vals = np.concatenate([d.values.flatten() for d in data_dict.values()])
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals) == 0:
        print(f"Warning: No valid data for {metric_name} in {period}")
        return

    custom_limits = get_custom_limits(metric_name, 'spatial')
    custom_levels = None
    if custom_limits is not None:
        if len(custom_limits) == 2:
            vmin, vmax = custom_limits[0], custom_limits[1]
        elif len(custom_limits) > 2:
            custom_levels = custom_limits
            vmin, vmax = custom_levels[0], custom_levels[-1]
    else:
        if use_robust_limits():
            local_vmin, local_vmax = np.percentile(all_vals, 2), np.percentile(all_vals, 98)
        else:
            local_vmin, local_vmax = all_vals.min(), all_vals.max()
        vmin, vmax = get_global_limits(local_vmin, local_vmax, region, save_path, period)
    
    if levels is None:
        low_metric = metric_name.lower()
        if "error" in low_metric or "bias" in low_metric or "difference" in low_metric:
            if custom_levels is not None:
                levels = custom_levels
            else:
                max_abs = max(abs(vmin), abs(vmax))
                if max_abs > 100:
                    levels = [-200, -100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 200]
                elif max_abs > 30:
                    levels = BIAS_EXTREME_LEVELS
                elif max_abs > 5:
                    levels = [-20, -10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, 20]
                else:
                    levels = BIAS_LEVELS
            cmap = ListedColormap(BIAS_RDBU_WHITE)
            norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, extend='both')

        elif "freq" in low_metric or "nbevents" in low_metric or "r95p" in low_metric or "r99p" in low_metric:
            if custom_levels is not None:
                levels = custom_levels
            else:
                if vmax <= 5:
                    levels = [0, 0.5, 1, 1.5, 2, 3, 4, 5]
                elif vmax <= 10:
                    levels = [0, 1, 2, 3, 4, 5, 6, 8, 10]
                elif vmax <= 20:
                    levels = [0, 2, 4, 6, 8, 10, 15, 20]
                elif vmax <= 30:
                    levels = [0, 5, 10, 15, 20, 25, 30]
                else:
                    levels = FREQ_LEVELS
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "custom_freq",
                ["white", "lightyellow", "gold", "darkorange", "crimson", "purple", "darkmagenta"],
                N=256
            )
            norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, extend='max')
        elif "cdd" in low_metric:
            if custom_levels is not None:
                levels = custom_levels
            else:
                if vmax <= 50:
                    levels = [0, 5, 10, 15, 20, 30, 40, 50]
                elif vmax <= 100:
                    levels = [0, 10, 20, 30, 40, 60, 80, 100]
                elif vmax <= 150:
                    levels = [0, 20, 40, 60, 80, 100, 120, 150]
                else:
                    levels = CDD_LEVELS
            cmap = plt.get_cmap("YlOrBr")
            norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, extend='max')
        elif "r01" in low_metric:
            if custom_levels is not None:
                levels = custom_levels
            else:
                if vmax <= 20:
                    levels = [0, 2, 4, 6, 8, 10, 15, 20]
                elif vmax <= 50:
                    levels = [0, 5, 10, 15, 20, 30, 40, 50]
                else:
                    levels = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            cmap = plt.get_cmap("YlGnBu")
            norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, extend='max')
        elif low_metric in ["mean", "rmse", "precipitation", "r95", "r99"]:
            if custom_levels is not None:
                levels = custom_levels
            else:
                if vmax <= 10:
                    levels = PRECIP_MEAN_LEVELS
                elif vmax <= 30:
                    levels = [0, 1, 2, 5, 10, 15, 20, 25, 30]
                elif vmax <= 60:
                    levels = [0, 2, 5, 10, 20, 30, 40, 50, 60]
                elif vmax <= 100:
                    levels = [0, 5, 10, 20, 30, 40, 60, 80, 100]
                else:
                    levels = PRECIP_EXTREME_LEVELS if low_metric in ["r95", "r99"] else PRECIP_MEAN_LEVELS
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "custom_precip",
                ["white", "lightblue", "darkblue", "darkgreen", "lightgreen", "yellow", "orange", "red"],
                N=256
            )
            norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, extend='max')
        elif "correlation" in low_metric or "rocss" in low_metric:
            cmap = plt.get_cmap("RdYlGn") if "rocss" in low_metric else plt.get_cmap("RdBu_r")
            if custom_levels is not None:
                levels = custom_levels
            else:
                if "correlation" in low_metric:
                    levels = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                else:
                    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, extend='neither')
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap("viridis")
    else:
        norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, extend='max' if levels[0] >= 0 else 'both')

    # Data dict keys: Expect reference first (Rule 5.2)
    for i, (model_name, data) in enumerate(data_dict.items()):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection=ccrs.PlateCarree())
        
        extent = [data.lon.min(), data.lon.max(), data.lat.min(), data.lat.max()]
        im = ax.imshow(data, extent=extent, cmap=cmap, norm=norm, origin="upper", transform=ccrs.PlateCarree())
        
        ax.coastlines(resolution='10m', color='0.3', linewidth=0.8)
        morocco.plot(ax=ax, edgecolor="black", linewidth=1.2, transform=ccrs.PlateCarree())
        
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}
        
        # Summary statistics in title (Min, Mean, Max)
        spatial_min = float(data.min())
        spatial_mean = float(data.mean())
        spatial_max = float(data.max())
        title_str = f"{model_name}\nMin: {spatial_min:.2f} | Mean: {spatial_mean:.2f} | Max: {spatial_max:.2f} {unit}"
        ax.set_title(title_str, fontsize=10, pad=10, fontweight='semibold')
        ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]])

    # Shared colorbar (Rule 3.1)
    if n_models == 1:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', extend='both', ticks=levels)
        plt.subplots_adjust(right=0.88, top=0.88)
    else:
        cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both', ticks=levels)
        plt.subplots_adjust(bottom=0.2 / nrows, top=0.88, wspace=0.15)
    
    cbar.set_label(f"{metric_name.upper()} ({unit})", fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved scientific spatial map: {save_path}")
    else:
        plt.show()

def plot_spatial_summary(data_dict, label, bounds, cmap_val, shapefile=None, save_path=None, fig_title=None, region=None):
    """
    Generates a multi-panel spatial comparison plot for multiple models.
    
    Inputs:
        data_dict (dict): Keys are model names, values are 2D spatial xr.DataArrays.
        label (str): Label for the shared colorbar (e.g., 'Precipitation mm/day').
        bounds (list): Numeric boundaries for discrete colormap levels.
        cmap_val (str/Colormap): Colormap name or object.
        shapefile (str): Path to the Morocco shapefile for border plotting.
        save_path (str, optional): Path to save the figure. If None, displays it.
        fig_title (str, optional): Main title for the entire figure.
        
    Outputs:
        None (Displays or saves a matplotlib figure).
        
    How it works:
        1. Loads the Morocco boundary from the shapefile.
        2. Dynamically creates a multi-column subplot layout based on the number of models.
        3. For each model: Plots the 2D data, adds coastlines, and overlays the Morocco border.
        4. Calculates and adds the spatial mean to each subplot title.
        5. Adds a single horizontal colorbar at the bottom.
        6. Adds a global title (suptitle) if fig_title is provided.
    """
    if region is None:
        region = get_current_region()
    if shapefile is None or shapefile == DEFAULT_SHAPEFILE:
        shapefile = get_shapefile(region)
        
    morocco_gdf = gpd.read_file(shapefile).dissolve()
    morocco = morocco_gdf.boundary
    map_bounds = morocco_gdf.total_bounds
    if isinstance(cmap_val, str):
        cmap = plt.get_cmap(cmap_val)
    else:
        cmap = cmap_val
    
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    keys = list(data_dict.keys())
    n = len(keys)
    
    # Adjust top margin if fig_title is present
    top_margin = 0.85 if fig_title else 0.95
    fig, axs = plt.subplots(1, n, figsize=(5.5 * n, 7), 
                            subplot_kw=dict(projection=ccrs.PlateCarree()), dpi=300)
    plt.subplots_adjust(top=top_margin)
    
    if n == 1: axs = [axs]
    
    if fig_title:
        metadata_str = get_title_metadata(fig_title, "")
        fig.suptitle(fig_title + metadata_str, fontsize=14, fontweight='bold', y=0.95)
    
    for j, ax in enumerate(axs):
        key = keys[j]
        data = data_dict[key]
        extent = [data.lon.min(), data.lon.max(), data.lat.min(), data.lat.max()]
        img = ax.imshow(data, extent=extent, cmap=cmap, norm=norm, origin="upper")
        ax.coastlines(resolution='10m', linewidth=0.5)
        morocco.plot(ax=ax, edgecolor="black", linewidth=1, transform=ccrs.PlateCarree())
        s_min, s_mean, s_max = float(data.min()), float(data.mean()), float(data.max())
        ax.set_title(f'{key}\nMin: {s_min:.2f} | Mean: {s_mean:.2f} | Max: {s_max:.2f}', fontsize=12)
        ax.set_extent([map_bounds[0], map_bounds[2], map_bounds[1], map_bounds[3]])
        
    if n == 1:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(img, cax=cbar_ax, orientation='vertical', extend='both', ticks=bounds, label=label)
        plt.subplots_adjust(right=0.88)
    else:
        cbar_ax = fig.add_axes([0.2, 0.15, 0.6, 0.03])
        fig.colorbar(img, cax=cbar_ax, orientation='horizontal', extend='both', ticks=bounds, label=label)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close(fig)

def plot_spatial_bias(data_dict, label='Bias (mm/day)', bounds=None, shapefile=None, save_path=None, fig_title=None, region=None):
    """
    Generates a multi-panel spatial comparison plot for error/bias maps (Model - Observation).
    Uses a divergent colormap centering at 0 (RdBu_r).
    
    Inputs:
        data_dict (dict): Keys are model names, values are 2D spatial xr.DataArrays (Model - Obs).
        label (str): Label for the shared colorbar including the unit.
        bounds (list, optional): Numeric boundaries for discrete colormap levels.
        shapefile (str): Path to the Morocco shapefile for border plotting.
        save_path (str, optional): Path to save the figure. If None, displays it.
        fig_title (str, optional): Main title for the entire figure.
        
    Outputs:
        None (Displays or saves a matplotlib figure).
        
    How it works:
        1. Loads the Morocco boundary from the provided shapefile.
        2. Uses the 'RdBu_r' diverging colormap.
        3. Iterates through the models, creating a subplot panel for each.
        4. Overlays physical features (coastlines) and national borders.
        5. Calculates the mean bias for each model and displays it in the panel title.
        6. Adds a centered, horizontal colorbar with the specified label and discrete bounds if provided.
    """
    if region is None:
        region = get_current_region()
    if shapefile is None or shapefile == DEFAULT_SHAPEFILE:
        shapefile = get_shapefile(region)
        
    morocco_gdf = gpd.read_file(shapefile).dissolve()
    morocco = morocco_gdf.boundary
    map_bounds = morocco_gdf.total_bounds
    
    if bounds is None:
        # Default bias bounds as requested: -2 to 2 with white in [-0.25, 0.25]
        bounds = [-2, -1.5, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.5, 2]
    cmap = ListedColormap(BIAS_RDBU_WHITE)

    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    keys = list(data_dict.keys())
    n = len(keys)
    
    top_margin = 0.85 if fig_title else 0.95
    fig, axs = plt.subplots(1, n, figsize=(5.5 * n, 7), 
                            subplot_kw=dict(projection=ccrs.PlateCarree()), dpi=300)
    plt.subplots_adjust(top=top_margin)
    
    if n == 1: axs = [axs]
    
    if fig_title:
        metadata_str = get_title_metadata(fig_title, "")
        fig.suptitle(fig_title + metadata_str, fontsize=14, fontweight='bold', y=0.95)
    
    for j, ax in enumerate(axs):
        key = keys[j]
        data = data_dict[key]
        extent = [data.lon.min(), data.lon.max(), data.lat.min(), data.lat.max()]
        
        im = ax.imshow(data, extent=extent, cmap=cmap, norm=norm, origin="upper")
        ax.coastlines(resolution='10m', linewidth=0.5)
        morocco.plot(ax=ax, edgecolor="black", linewidth=1, transform=ccrs.PlateCarree())
        s_min, s_mean, s_max = float(data.min()), float(data.mean()), float(data.max())
        ax.set_title(f'{key} Error\nMin: {s_min:.2f} | Mean Bias: {s_mean:.2f} | Max: {s_max:.2f}', fontsize=12)
        ax.set_extent([map_bounds[0], map_bounds[2], map_bounds[1], map_bounds[3]])
        
    if n == 1:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, orientation='vertical', extend='both', ticks=bounds, label=label)
        plt.subplots_adjust(right=0.88)
    else:
        cbar_ax = fig.add_axes([0.2, 0.15, 0.6, 0.03])
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both', ticks=bounds, label=label)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def compute_generic_metrics(dataset, ref, metric_func, model_name, suffix, output_dir, strategy, corr_strategy, return_by_year, compute_periods=None, **kwargs):
    """
    An orchestration engine that computes a metric across all standard seasons.
    
    Inputs:
        dataset (xr.DataArray): The model prediction data.
        ref (xr.DataArray): The reference (truth) data.
        metric_func (function): The calculation function (e.g., compute_bias_flagged).
        model_name (str): Name of the model for logging/saving.
        suffix (str): Additional identifying tag for filenames.
        output_dir (str): Where to save results.
        strategy (str): Calculation strategy to pass to metric_func.
        corr_strategy (str): Correlation strategy identifier.
        return_by_year (bool): Whether results stay in (year, lat, lon) format.
        compute_periods (list): Specific periods to calculate (e.g. ['Annual']).
        **kwargs: Extra parameters passed directly to metric_func.
        
    Outputs:
        tuple (dict, pd.DataFrame): 
            - Dict of seasonal spatial maps.
            - DataFrame of spatial means for each season.
            
    How it works:
        1. Extracts month information from the dataset.
        2. Loops through standard seasons (DJF, MAM, JJA, SON) and Annual.
        3. Filters data for each season and calls metric_func on the subset.
        4. Saves each seasonal map as a NetCDF file.
        5. Compiles spatial averages into a summary DataFrame.
    """
    if dataset is None or dataset.time.size == 0:
        print(f"Warning: Dataset for {model_name} is empty for the requested period. Skipping.")
        return {}, pd.DataFrame()

    if ref is not None:
        if 'time' in dataset.coords:
            dataset['time'] = dataset['time'].dt.floor('D')
        if 'time' in ref.coords:
            ref['time'] = ref['time'].dt.floor('D')
        
        # Intersect time manually to avoid dropping lat/lon due to float precision differences
        common_time = np.intersect1d(dataset.time.values, ref.time.values)
        dataset = dataset.sel(time=common_time)
        ref = ref.sel(time=common_time)

    _, months = get_time_info(dataset['time'].values)
    seasonal_maps = {}
    domain_averages = {}

    if compute_periods is None:
        config_path = os.environ.get("POSTPROC_MASTER_CONFIG", os.path.join(PROJECT_ROOT, "config.yaml"))
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                global_cfg = yaml.safe_load(f)
                if global_cfg and 'parameters' in global_cfg:
                    compute_periods = global_cfg['parameters'].get('compute_periods', global_cfg['parameters'].get('plot_periods', list(SEASONS.keys())))
    if compute_periods is None:
        compute_periods = list(SEASONS.keys())

    for season, season_months in SEASONS.items():
        if season not in compute_periods:
            continue
        idx = np.isin(months, season_months)
        data_s = dataset.isel(time=idx)
        ref_s = ref.isel(time=idx) if ref is not None else None

        # Call the specific metric function with extra kwargs
        metric_map = metric_func(data_s, ref_s, strategy=strategy, return_by_year=return_by_year, **kwargs)
        
        metric_name = metric_func.__name__.replace('compute_', '').replace('_flagged', '')
        metrics_dict = {metric_name: metric_map}

        prefix = f"{model_name}_{suffix}_strategy_{strategy}_corr_{corr_strategy}"
        save_metrics_to_netcdf(metrics_dict, prefix, output_dir, tag=season)

        seasonal_maps[season] = metrics_dict
        domain_averages[season] = {
            f'{metric_name.upper()}': float(metric_map.mean().values) if metric_map is not None else np.nan
        }

    df = pd.DataFrame(domain_averages).T
    return seasonal_maps, df

def load_metric_results(model_paths, period="Annual"):
    """
    Loads saved metric NetCDF files for multiple models for a specific period.
    
    Inputs:
        model_paths (dict): Keys are model names, values are file path templates with "PERIOD".
        period (str): The season/period to load (e.g., 'Annual').
        
    Outputs:
        dict: Keys are model names, values are loaded xr.DataArrays.
        
    How it works:
        1. Replaces 'PERIOD' in path templates with the selected period.
        2. Opens each file and automatically selects the first available data variable.
        3. Returns a consistent dictionary of arrays.
    """
    results = {}
    for model_name, path in model_paths.items():
        filename = path.replace("PERIOD", period)
        if os.path.exists(filename):
            ds = xr.open_dataset(filename)
            # Use the first data variable found
            var_name = list(ds.data_vars.keys())[0]
            results[model_name] = ds[var_name]
    return results

def get_regional_means(ds, shapefile=None):
    """
    Calculates the average value specifically within a geographic region defined by a shapefile.
    
    Inputs:
        ds (xr.DataArray): Input data (spatial map or temporal series).
        shapefile (str): Path to the region mask shapefile.
        
    Outputs:
        Single value OR (np.array, np.array): 
            - If 2D: Returns a single spatial mean.
            - If Temporal: Returns (array of means, time axis values).
            
    How it works:
        1. Loads shapefile and converts to a single boundary polygon.
        2. Creates a point-in-polygon mask by checking every pixel coordinate.
        3. If temporal data exists: Iterates through time, applying the mask to each year.
        4. Calculates np.nanmean on masked values to ignore ocean/outside areas.
    """
    if shapefile is None or shapefile == DEFAULT_SHAPEFILE:
        shapefile = get_shapefile(get_current_region())
    morocco = gpd.read_file(shapefile).unary_union
    # Check if we have year or time dimension
    dim = 'year' if 'year' in ds.dims else ('time' if 'time' in ds.dims else None)
    
    if dim:
        results = []
        for i in range(ds.sizes[dim]):
            data = ds.isel({dim: i})
            lon2d, lat2d = np.meshgrid(data["lon"], data["lat"])
            points = gpd.GeoSeries(gpd.points_from_xy(lon2d.ravel(), lat2d.ravel()), crs="EPSG:4326")
            mask = points.within(morocco).values.reshape(data.shape)
            results.append(np.nanmean(data.values[mask]))
        return np.array(results), ds[dim].values
    else:
        # Just spatial 2D
        lon2d, lat2d = np.meshgrid(ds["lon"], ds["lat"])
        points = gpd.GeoSeries(gpd.points_from_xy(lon2d.ravel(), lat2d.ravel()), crs="EPSG:4326")
        mask = points.within(morocco).values.reshape(ds.shape)
        return np.nanmean(ds.values[mask]), None

def plot_temporal_evolution(model_paths, metric_name, period="Annual", shapefile=None, save_path=None, unit="mm/day"):
    """
    Plots a multi-line graph showing regional average evolution over time.
    """
    current_region = get_current_region()
    if shapefile is None or shapefile == DEFAULT_SHAPEFILE:
        shapefile = get_shapefile(current_region)
    plt.figure(figsize=(10, 6), dpi=300)
    fallback_colors = ['#F4A261', '#8E44AD', '#E76F51']
    
    local_min, local_max = np.inf, -np.inf
    
    for i, (model_name, path) in enumerate(model_paths.items()):
        filename = path.replace("PERIOD", period)
        if os.path.exists(filename):
            ds = xr.open_dataset(filename)
            var = list(ds.data_vars.values())[0]
            means, time_axis = get_regional_means(var, shapefile)
            
            if time_axis is None:
                print(f"Warning: No time/year dimension for {model_name}. Skipping temporal plot.")
                continue
                
            local_min = min(local_min, np.nanmin(means))
            local_max = max(local_max, np.nanmax(means))
                
            color = GLOBAL_MODEL_COLORS.get(model_name.upper(), fallback_colors[i % len(fallback_colors)])
            plt.plot(time_axis, means, label=model_name, marker='o', markersize=5, 
                     linewidth=2, color=color, alpha=0.85)
    
    custom_limits = get_custom_limits(metric_name, 'temporal')
    if custom_limits is not None:
        plt.ylim(custom_limits[0], custom_limits[-1])
    else:
        g_min, g_max = get_global_evolution_limits(current_region, save_path, period)
        if g_min is not None and g_max is not None:
            plt.ylim(min(local_min, g_min), max(local_max, g_max))
    
    plt.xlabel("Year", fontsize=12)
    plt.ylabel(f"{metric_name.upper()} ({unit})", fontsize=12)
    title_str = f"Temporal Evolution: {metric_name.upper()} ({period})"
    title_str += get_title_metadata(metric_name, period)
    plt.title(title_str, fontsize=11, fontweight='bold', pad=15)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc='best')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"Saved evolution plot: {save_path}")
    else:
        plt.show()

def plot_metric_boxplot(model_paths, metric_name, period="Annual", shapefile=None, save_path=None):
    """
    Visualizes the statistical distribution (spread) of metric values across time and space.
    
    Inputs:
        model_paths (dict): Paths to saved metric NetCDF files.
        metric_name (str): Name of the metric for labeling.
        period (str): The season to plot.
        shapefile (str): Path to the regional shapefile.
        save_path (str, optional): Target save location.
        
    Outputs:
        None (Displays or saves a seaborn figure).
        
    How it works:
        1. Loads the Morocco boundary from the shapefile.
        2. For each model, extracts every single pixel that falls inside the boundary for all years.
        3. Collects these thousands of values into a long-form DataFrame.
        4. Uses seaborn.boxplot to show the median, spread (IQR), and full range of values.
    """
    import seaborn as sns
    all_data = []
    if shapefile is None or shapefile == DEFAULT_SHAPEFILE:
        shapefile = get_shapefile(get_current_region())
    morocco = gpd.read_file(shapefile).unary_union

    for model_name, path in model_paths.items():
        filename = path.replace("PERIOD", period)
        if os.path.exists(filename):
            ds = xr.open_dataset(filename)
            var = list(ds.data_vars.values())[0]
            
            lon2d, lat2d = np.meshgrid(var["lon"], var["lat"])
            points = gpd.GeoSeries(gpd.points_from_xy(lon2d.ravel(), lat2d.ravel()), crs="EPSG:4326")
            mask = points.within(morocco).values.reshape((var.sizes['lat'], var.sizes['lon']))
            
            data_values = var.values
            for i in range(var.sizes.get('year', 1)):
                frame = data_values[i, :, :] if 'year' in var.dims else data_values
                valid_vals = frame[mask]
                valid_vals = valid_vals[~np.isnan(valid_vals)]
                all_data.extend([(model_name, v) for v in valid_vals])

    if not all_data:
        print(f"Warning: No valid data found for boxplot in {period}")
        return

    df = pd.DataFrame(all_data, columns=["Model", metric_name.upper()])
    
    if use_robust_limits():
        local_min = df[metric_name.upper()].quantile(0.02)
        local_max = df[metric_name.upper()].quantile(0.98)
    else:
        local_min = df[metric_name.upper()].min()
        local_max = df[metric_name.upper()].max()
        
    g_min, g_max = get_global_limits(local_min, local_max, get_current_region(), save_path, period)
    
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Consistent vibrant colors for models mapping to user requested colors
    fallback_colors = ['#F4A261', '#8E44AD', '#E76F51']
    palette = {name: GLOBAL_MODEL_COLORS.get(name.upper(), fallback_colors[i % len(fallback_colors)]) 
               for i, name in enumerate(model_paths.keys())}

    sns.boxplot(data=df, x="Model", y=metric_name.upper(), palette=palette, 
                hue="Model", legend=False, showfliers=False, width=0.5, linewidth=1.2)
    
    # Annotate with mean as markers (Rule 3.4)
    sns.pointplot(data=df, x="Model", y=metric_name.upper(), estimator=np.mean, 
                  linestyle='none', color="darkred", markers="D", native_scale=True)

    custom_limits = get_custom_limits(metric_name, 'boxplot')
    if custom_limits is not None:
        plt.ylim(custom_limits[0], custom_limits[-1])
    elif g_min is not None and g_max is not None:
        padding = (g_max - g_min) * 0.05
        plt.ylim(min(local_min, g_min) - padding, max(local_max, g_max) + padding)

    title_str = f"Distribution Analysis: {metric_name.upper()} ({period})"
    title_str += get_title_metadata(metric_name, period)
    plt.title(title_str, fontsize=11, fontweight='bold', pad=15)
    plt.xlabel("Model Configuration", fontsize=11)
    plt.ylabel(f"{metric_name.upper()} Range", fontsize=11)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"Saved boxplot distribution: {save_path}")
    else:
        plt.show()

def plot_monthly_cycle(datasets_dict, region=None, shapefile=None, save_path=None):
    """
    Plots the climatological annual cycle (monthly averages).
    
    Inputs:
        datasets_dict (dict): Dictionary of datasets where keys are model names.
        region (str, optional): Region identifier for potential filtering.
        shapefile (str, optional): Path to region shapefile.
        save_path (str, optional): Where to save the plot.
    """
    if region is None:
        region = get_current_region()
    if shapefile is None or shapefile == DEFAULT_SHAPEFILE:
        shapefile = get_shapefile(region)
        
    plt.figure(figsize=(10, 6), dpi=300)
    fallback_colors = ['#F4A261', '#8E44AD', '#E76F51']
    
    for i, (name, ds) in enumerate(datasets_dict.items()):
        # Calculate monthly means
        monthly_ds = ds.groupby('time.month').mean(dim='time')
        
        # Domain mean for each month
        means = []
        for m in range(1, 13):
            m_data = monthly_ds.sel(month=m)
            m_mean, _ = get_regional_means(m_data, shapefile)
            means.append(m_mean)
        
        color = GLOBAL_MODEL_COLORS.get(name.upper(), fallback_colors[i % len(fallback_colors)])
        plt.plot(range(1, 13), means, label=name, marker='o', linewidth=2, color=color)

    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Precipitation (mm/day)", fontsize=12)
    title_str = f"Annual Cycle: Monthly Climatology ({region})"
    title_str += get_title_metadata("mean", "Monthly Climatology")
    plt.title(title_str, fontsize=11, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"Saved monthly cycle: {save_path}")
    else:
        plt.show()

def plot_intensity_distribution_log(datasets_dict, region=None, shapefile=None, save_path=None, threshold=0.1):
    """
    Plots the Probability Density Function (PDF) of daily precipitation intensities 
    using a logarithmic scale for both X and Y axes.
    """
    if region is None:
        region = get_current_region()
    if shapefile is None or shapefile == DEFAULT_SHAPEFILE:
        shapefile = get_shapefile(region)
        
    plt.figure(figsize=(10, 6), dpi=300)
    line_colors = ['#E63946', '#457B9D', '#1D3557', '#2A9D8F', '#F4A261', '#8E44AD']
    
    for i, (name, ds) in enumerate(datasets_dict.items()):
        lon2d, lat2d = np.meshgrid(ds["lon"], ds["lat"])
        points = gpd.GeoSeries(gpd.points_from_xy(lon2d.ravel(), lat2d.ravel()), crs="EPSG:4326")
        morocco = gpd.read_file(shapefile).unary_union
        mask = points.within(morocco).values.reshape((ds.sizes['lat'], ds.sizes['lon']))
        
        vals = ds.values[:, mask].flatten()
        vals = vals[~np.isnan(vals)]
        vals = vals[vals >= threshold]
        
        color = GLOBAL_MODEL_COLORS.get(name.upper(), line_colors[i % len(line_colors)])
        if len(vals) > 1:
            bins = np.logspace(np.log10(threshold), np.log10(max(vals) if max(vals) > threshold else threshold+10), 50)
            plt.hist(vals, bins=bins, histtype='step', label=name, color=color, linewidth=2, density=True)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Daily Intensity (mm/day)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    title_str = f"Intensity Distribution Log-Log ({region})"
    title_str += get_title_metadata("intensity", "All Days")
    plt.title(title_str, fontsize=11, fontweight='bold', pad=15)
    plt.grid(True, which="both", linestyle='--', alpha=0.3)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"Saved log intensity distribution: {save_path}")
    else:
        plt.show()

def plot_intensity_distribution_linear(datasets_dict, region=None, shapefile=None, save_path=None, threshold=0.1):
    """
    Plots the Probability Density Function (PDF) of daily precipitation intensities 
    using Kernel Density Estimation (KDE) and linear axes (Matches pdf.py style).
    """
    from scipy.stats import gaussian_kde
    if region is None:
        region = get_current_region()
    if shapefile is None or shapefile == DEFAULT_SHAPEFILE:
        shapefile = get_shapefile(region)
        
    plt.figure(figsize=(10, 6), dpi=300)
    line_colors = ['#E63946', '#457B9D', '#1D3557', '#2A9D8F', '#F4A261', '#8E44AD']
    
    # Pre-load Morocco mask once
    morocco = gpd.read_file(shapefile).unary_union
    x_range = np.linspace(0, 30, 200) # Standard range for bulk distribution
    
    for i, (name, ds) in enumerate(datasets_dict.items()):
        lon2d, lat2d = np.meshgrid(ds["lon"], ds["lat"])
        points = gpd.GeoSeries(gpd.points_from_xy(lon2d.ravel(), lat2d.ravel()), crs="EPSG:4326")
        mask = points.within(morocco).values.reshape((ds.sizes['lat'], ds.sizes['lon']))
        
        vals = ds.values[:, mask].flatten()
        vals = vals[~np.isnan(vals)]
        vals = vals[vals >= threshold]
        
        color = GLOBAL_MODEL_COLORS.get(name.upper(), line_colors[i % len(line_colors)])
        
        if len(vals) > 1:
            # KDE calculation
            kde = gaussian_kde(vals)
            pdf_vals = kde(x_range)
            
            plt.plot(x_range, pdf_vals, color=color, label=name, linewidth=2)
            
            # Vertical line for mean
            dist_mean = np.mean(vals)
            plt.axvline(dist_mean, color=color, linestyle='--', alpha=0.6, linewidth=1.5)

    plt.xlabel("Daily Intensity (mm/day)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    title_str = f"Intensity Distribution Linear KDE ({region})"
    title_str += get_title_metadata("intensity", "All Days")
    plt.title(title_str, fontsize=11, fontweight='bold', pad=15)
    plt.xlim(0, 30)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"Saved linear intensity distribution: {save_path}")
    else:
        plt.show()
