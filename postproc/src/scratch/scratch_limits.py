# Author: M. El Aabaribaoune (@um6p)
import os
import glob
import xarray as xr
import pandas as pd
import numpy as np

def analyze_metric(metric_dir, is_error=False):
 print(f"\n--- Analyzing {metric_dir} ---")
 results_dir = os.path.join(metric_dir, "results")
 if not os.path.exists(results_dir):
 return

 nc_files = glob.glob(os.path.join(results_dir, "*.nc"))
 csv_files = glob.glob(os.path.join(results_dir, "*.csv"))

 spatial_mins, spatial_maxs = [], []
 temporal_mins, temporal_maxs = [], []

 def is_valid(f):
 base = os.path.basename(f).lower()
 return "glm" not in base and "lmdz_35" not in base

 for f in filter(is_valid, nc_files):
 ds = xr.open_dataset(f)
 for var in ds.data_vars:
 data = ds[var].values
 data = data[~np.isnan(data)]
 if len(data) > 0:
 p1, p99 = np.percentile(data, 1), np.percentile(data, 99)
 spatial_mins.append(p1)
 spatial_maxs.append(p99)
 ds.close()

 for f in filter(is_valid, csv_files):
 df = pd.read_csv(f)
 for col in df.columns:
 if col.lower() not in ["year", "model", "period"]:
 # try converting to numeric
 data = pd.to_numeric(df[col], errors='coerce').dropna().values
 if len(data) > 0:
 temporal_mins.append(data.min())
 temporal_maxs.append(data.max())

 if spatial_mins:
 s_min, s_max = min(spatial_mins), max(spatial_maxs)
 if is_error:
 if "rmse" in metric_dir:
 s_min = 0
 if "bias" in metric_dir:
 m = max(abs(s_min), abs(s_max))
 s_min, s_max = -m, m
 print(f"Spatial suggested limits: [{s_min:.2f}, {s_max:.2f}]")
 
 if temporal_mins:
 t_min, t_max = min(temporal_mins), max(temporal_maxs)
 if is_error:
 if "rmse" in metric_dir:
 t_min = 0
 if "bias" in metric_dir:
 m = max(abs(t_min), abs(t_max))
 t_min, t_max = -m, m
 print(f"Temporal suggested limits: [{t_min:.2f}, {t_max:.2f}]")

base_path = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/results/vits_glm_vs_cnn_lmdz250/1979-01-01_2014-12-31/allmorr/test"
for d in ["bias", "rmse", "correlation", "cdd", "r95", "r95_nbEvents_freq"]:
 is_err = d in ["bias", "rmse"]
 analyze_metric(os.path.join(base_path, d), is_err)
