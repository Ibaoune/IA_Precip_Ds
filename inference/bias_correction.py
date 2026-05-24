# Author: M. El Aabaribaoune (@um6p)
import gc
import xarray as xr
import numpy as np

def scaling_delta_mapping(gcm_full, gcm_hist, obs_hist):
 """
 Implements the Signal-Preserving Scaling Delta Mapping (SDM)
 from Baño-Medina et al. (2022).

 Logic:
 1. Signal = Mean_m(G_full) - Mean_m(G_hist)
 2. Detrended = G_full - Signal
 3. Bias Correct: (Detrended - Mean_m(G_hist)) / Std_m(G_hist) * Std_m(Obs_hist) + Mean_m(Obs_hist)
 4. Re-trend: Corrected + Signal
 """

 # 1. Calculate Monthly stats — load one source at a time to cap peak RAM.
 # gcm_hist: compute both stats before loading obs_hist.
 gcm_hist_grouped = gcm_hist.groupby("time.month")
 mean_gcm_hist = gcm_hist_grouped.mean(dim="time").compute()
 std_gcm_hist = gcm_hist_grouped.std(dim="time").compute()
 del gcm_hist_grouped, gcm_hist
 gc.collect()

 # obs_hist: gcm_hist raw data can now be freed before this load.
 obs_hist_grouped = obs_hist.groupby("time.month")
 mean_obs_hist = obs_hist_grouped.mean(dim="time").compute()
 std_obs_hist = obs_hist_grouped.std(dim="time").compute()
 del obs_hist_grouped, obs_hist
 gc.collect()

 # gcm_full monthly mean (needed for the climate-change signal).
 gcm_full_grouped = gcm_full.groupby("time.month")
 mean_gcm_full = gcm_full_grouped.mean(dim="time").compute()
 del gcm_full_grouped
 gc.collect()

 # 2. Apply Signal-Preserving Correction
 def _apply_sdm_signal_preserved(group):
 month = group.time.dt.month[0].values
 
 # Monthly parameters
 mu_gh = mean_gcm_hist.sel(month=month)
 sig_gh = std_gcm_hist.sel(month=month)
 mu_oh = mean_obs_hist.sel(month=month)
 sig_oh = std_obs_hist.sel(month=month)
 mu_gf = mean_gcm_full.sel(month=month)
 
 # Calculate Signal (Trend)
 signal = mu_gf - mu_gh
 
 # Detrend anomalies
 detrended = group - signal
 
 # Bias Correct anomalies to Observation level
 corrected_anomalies = (detrended - mu_gh) / (sig_gh + 1e-8) * sig_oh + mu_oh
 
 # Add Signal back
 final = corrected_anomalies + signal
 return final

 gcm_corrected = gcm_full.groupby("time.month").map(_apply_sdm_signal_preserved)
 
 return gcm_corrected

def standardize_predictors(ds, mean_ref, std_ref):
 """
 Standardize predictors based on a reference mean and std.
 """
 return (ds - mean_ref) / (std_ref + 1e-8)
