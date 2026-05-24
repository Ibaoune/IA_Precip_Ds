# Author: M. El Aabaribaoune (@um6p)
import gc
import os
import sys
import yaml
import torch
import xarray as xr
import numpy as np

# Local imports
from config import load_config
from data_loading import load_datasets
from utils import vprint, load_model, set_verbose, _build_model as get_model
from bias_correction import scaling_delta_mapping, standardize_predictors




def perform_single_prediction(cfg, scenario=None, abs_results_dir=None):
 """
 Load -> Interpolate -> Bias Correction -> Standardize -> Predict -> Save
 """
 if scenario:
 vprint(f"\n--- Starting Scenario: {scenario['name']} ---")
 cfg.src = scenario["src"]
 cfg.start_date_test = scenario["start"]
 cfg.end_date_test = scenario["end"]
 cfg.folder = scenario.get("folder")
 cfg.bc_reference_folder = scenario.get("bc_reference_folder", cfg.bc_reference_folder)
 apply_bc = scenario.get("bias_correction", False)
 else:
 apply_bc = False

 # Snapshot cfg fields that will be temporarily mutated
 orig_src = cfg.src
 orig_folder = cfg.folder
 orig_start = cfg.start_date_test
 orig_end = cfg.end_date_test
 orig_ref_folder = cfg.bc_reference_folder

 def restore_cfg():
 cfg.src = orig_src
 cfg.folder = orig_folder
 cfg.start_date_test = orig_start
 cfg.end_date_test = orig_end
 cfg.bc_reference_folder = orig_ref_folder

 # 1 & 2. Load + Interpolate
 vprint(f"Flow Step 1 & 2: Loading and Interpolating {cfg.src} predictors...")
 datasets = load_datasets(cfg)
 X = datasets[0].sel(time=slice(cfg.start_date_test, cfg.end_date_test))
 lon_out, lat_out, time_test = datasets[5], datasets[6], datasets[8]

 # 3. Bias Correction + Standardization reference
 if apply_bc or cfg.norm_mode == "gridbox":
 vprint(f"Loading historical references (ERA5/GCM {cfg.ref_start} to {cfg.ref_end})...")

 # Use the absolute results_dir that was fixed in main() — cfg.results_dir may
 # have been reset to the relative training-config value by Config.__setattr__
 results_dir = abs_results_dir if abs_results_dir else cfg.results_dir
 cache_dir = os.path.join(results_dir, cfg.experiment, "stats")
 os.makedirs(cache_dir, exist_ok=True)
 mean_path = os.path.join(cache_dir, f"mean_era5_{cfg.ref_start}_{cfg.ref_end}_{cfg.norm_mode}.nc")
 std_path = os.path.join(cache_dir, f"std_era5_{cfg.ref_start}_{cfg.ref_end}_{cfg.norm_mode}.nc")

 # Load ERA5 historical whenever stats are not cached or BC is needed
 if not (os.path.exists(mean_path) and os.path.exists(std_path)) or apply_bc:
 vprint(f"Loading ERA5 historical ({cfg.ref_start} to {cfg.ref_end})...")
 cfg.src = "era5"
 cfg.start_date_test, cfg.end_date_test = cfg.ref_start, cfg.ref_end
 X_era5_hist = load_datasets(cfg)[0].sel(time=slice(cfg.ref_start, cfg.ref_end))
 restore_cfg()

 if os.path.exists(mean_path) and os.path.exists(std_path):
 vprint(f"Loading cached ERA5 stats...")
 mean_ref = xr.open_dataarray(mean_path)
 std_ref = xr.open_dataarray(std_path)
 else:
 vprint("Computing and caching ERA5 stats...")
 mean_ref = X_era5_hist.mean(dim="time") if cfg.norm_mode == "gridbox" else X_era5_hist.mean()
 std_ref = X_era5_hist.std(dim="time") if cfg.norm_mode == "gridbox" else X_era5_hist.std()
 mean_ref.to_netcdf(mean_path)
 std_ref.to_netcdf(std_path)
 vprint(f"Stats cached to {cache_dir}")

 if apply_bc:
 vprint("Flow Step 3: Applying Bias Correction (SDM)...")
 # GCM historical: use the reference folder (e.g. Present) instead of scenario folder
 cfg.src = orig_src
 cfg.folder = cfg.bc_reference_folder
 cfg.start_date_test, cfg.end_date_test = cfg.ref_start, cfg.ref_end
 X_gcm_hist = load_datasets(cfg)[0].sel(time=slice(cfg.ref_start, cfg.ref_end))
 restore_cfg()
 X = scaling_delta_mapping(X, X_gcm_hist, X_era5_hist)
 # Release large historical arrays immediately — they are no longer needed
 # and keeping them alongside X during standardization wastes peak RAM.
 del X_gcm_hist, X_era5_hist
 gc.collect()
 else:
 vprint("Flow Step 3: Skipping Bias Correction.")
 else:
 vprint("Flow Step 3: Skipping Bias Correction.")
 mean_ref, std_ref = 0.0, 1.0

 # 4. Standardization
 vprint("Flow Step 4: Standardizing predictors (Baseline: ERA5 Hist)...")
 X_std = standardize_predictors(X, mean_ref, std_ref)

 # 5. Load model (cached across scenarios) + Inference
 # Restore absolute results_dir before load_model — cfg may have been reset
 if abs_results_dir:
 cfg.results_dir = abs_results_dir
 cfg.exp_dir = os.path.join(abs_results_dir, cfg.experiment)

 if not hasattr(perform_single_prediction, "cached_model"):
 vprint("Flow Step 5: Preparing Model...")
 if cfg.model_type == "glm":
 model, _, _ = load_model(cfg, None)
 else:
 n_channels = X_std.shape[1]
 out_shape = (len(lat_out), len(lon_out))
 dummy_x = torch.empty(1, n_channels, X_std.shape[2], X_std.shape[3])
 dummy_y = torch.empty(1, 1, *out_shape)
 model_arch = get_model(cfg, dummy_x, dummy_y).to(cfg.device)
 model, _, _ = load_model(cfg, model_arch)
 model.eval()
 perform_single_prediction.cached_model = model

 model = perform_single_prediction.cached_model
 vprint("Flow Step 5: Running Inference...")

 preds = []
 chunk_size = 512
 if cfg.model_type == "glm":
 for i in range(0, X_std.shape[0], chunk_size):
 xb_np = X_std.isel(time=slice(i, i + chunk_size)).values.astype("float32")
 out = model.predict(xb_np, chunk_size=chunk_size)
 preds.append(out)
 preds_np = np.concatenate(preds, axis=0)
 else:
 with torch.no_grad():
 for i in range(0, X_std.shape[0], chunk_size):
 xb_np = X_std.isel(time=slice(i, i + chunk_size)).values.astype("float32")
 xb = torch.tensor(xb_np, dtype=torch.float32).to(cfg.device)
 out = model(xb)
 if cfg.loss_type == "bernoulli_gamma":
 if cfg.model_type == "vit":
 occurrence = out[:, 0, :, :]
 shape = out[:, 1, :, :]
 scale = out[:, 2, :, :]
 else:
 occurrence = torch.sigmoid(out[:, 0, :, :])
 shape = torch.exp(out[:, 1, :, :].clamp(-10, 7))
 scale = torch.exp(out[:, 2, :, :].clamp(-10, 7))
 precip = occurrence * (shape * scale)
 else:
 precip = out[:, 0, :, :]
 preds.append(precip.cpu())
 preds_np = torch.cat(preds, dim=0).numpy()

 # 6. Save
 out_dir = os.path.abspath(cfg.output_dir)
 bc_suffix = "_true" if apply_bc else "_false"
 save_name = f"{cfg.model_type}_{scenario['name']}{bc_suffix}.nc" if scenario else f"{cfg.model_type}_predictions{bc_suffix}.nc"
 out_nc = os.path.join(out_dir, save_name)
 os.makedirs(out_dir, exist_ok=True)

 ds_pred = xr.Dataset(
 {"precipitation": (["time", "lat", "lon"], preds_np)},
 coords={"time": time_test, "lat": lat_out, "lon": lon_out}
 )
 ds_pred["precipitation"].attrs["units"] = "mm/day"
 ds_pred.to_netcdf(out_nc)
 vprint(f"Flow Step 6: Results saved to {out_nc}")


def main():
 current_dir = os.path.dirname(os.path.abspath(__file__))
 if len(sys.argv) > 1:
 pred_config_file = os.path.abspath(sys.argv[1])
 else:
 pred_config_file = os.path.join(current_dir, "config.yaml")

 with open(pred_config_file, "r") as f:
 pred_cfg_dict = yaml.safe_load(f)

 full_cfg_dict = pred_cfg_dict

 from config import Config
 cfg = Config(full_cfg_dict, train_mode=False)

 train_config_path = pred_cfg_dict.get("prediction", {}).get("train_config_path", "")
 if train_config_path and os.path.exists(train_config_path):
 vprint(f"Loading hyperparameters from train config: {train_config_path}")
 with open(train_config_path, "r") as tc_file:
 lines = tc_file.readlines()
 
 yaml_lines = []
 parsing_yaml = False
 for line in lines:
 if line.startswith("---"):
 parsing_yaml = True
 continue
 if parsing_yaml:
 yaml_lines.append(line)
 if not yaml_lines:
 yaml_lines = lines
 
 train_cfg_dict = yaml.safe_load("".join(yaml_lines))
 if isinstance(train_cfg_dict, dict):
 keys_to_update = [
 "model_type", "loss_type", "norm_mode", "learning_rate", "epochs",
 "batch_size", "group_norm_enable", "group_norm_num_groups",
 "training_dropout_enable", "training_dropout_value",
 "emb_size", "patch_size", "num_layers", "num_heads", "dropout",
 "model_save_dir", "variables", "levels", "resolution", "interpolation_type",
 "variable", "target", "lon_min", "lon_max", "lat_min", "lat_max"
 ]
 for k in keys_to_update:
 if k in train_cfg_dict:
 setattr(cfg, k, train_cfg_dict[k])
 vprint("Hyperparameters updated successfully.")

 # Resolve absolute results_dir from prediction config — do NOT rely on cfg.models_dir
 # since Config may not expose nested prediction.models_dir as a flat attribute.
 raw_models_dir = pred_cfg_dict.get("prediction", {}).get("models_dir", "../results/")
 if not os.path.isabs(raw_models_dir):
 abs_results_dir = os.path.abspath(os.path.join(current_dir, raw_models_dir))
 else:
 abs_results_dir = raw_models_dir

 # 1. Start with the complex path built by Config class
 complex_model_dir = cfg.model_save_dir
 
 # 2. Build the simple flat path as a fallback
 flat_model_dir = os.path.join(abs_results_dir, cfg.experiment, "models")
 
 # Logic to select the best model directory
 # We check if the complex path exists and contains the model file
 model_filename = f"{cfg.model_type}_{cfg.variable}.pth"
 complex_model_exists = os.path.exists(os.path.join(complex_model_dir, model_filename))
 
 if not complex_model_exists:
 # Check for .pkl fallback for GLM
 pkl_filename = f"{cfg.model_type}_{cfg.variable}.pkl"
 complex_model_exists = os.path.exists(os.path.join(complex_model_dir, pkl_filename))

 if complex_model_exists:
 vprint(f"Using complex model directory: {complex_model_dir}")
 # cfg.model_save_dir is already set to complex_model_dir by Config class
 else:
 flat_model_exists = os.path.exists(os.path.join(flat_model_dir, model_filename))
 if not flat_model_exists:
 pkl_filename = f"{cfg.model_type}_{cfg.variable}.pkl"
 flat_model_exists = os.path.exists(os.path.join(flat_model_dir, pkl_filename))
 
 if flat_model_exists:
 vprint(f"Using flat model directory: {flat_model_dir}")
 cfg.model_save_dir = flat_model_dir
 cfg.exp_dir = os.path.join(abs_results_dir, cfg.experiment)
 else:
 vprint(f"WARNING: Model not found in complex or flat paths. Defaulting to complex path: {complex_model_dir}")

 vprint(f"Final absolute model directory: {os.path.abspath(cfg.model_save_dir)}")
 if hasattr(cfg, "model_path") and cfg.model_path:
 vprint(f"User-specified model_path detected: {cfg.model_path}")

 # Flatten remaining prediction-section paths that Config doesn't expose directly
 pred_section = pred_cfg_dict.get("prediction", {})
 cfg.output_dir = pred_section.get("output_dir", os.path.join(current_dir, "output"))
 cfg.bc_reference_folder = pred_section.get("bc_reference_folder", "")
 if not hasattr(cfg, "scenarios") or not cfg.scenarios:
 cfg.scenarios = pred_section.get("scenarios", [])

 set_verbose(cfg.verbose)
 vprint(f"=== Starting Prediction Pipeline for {cfg.experiment} ===")
 vprint(f"Models dir: {cfg.model_save_dir}")

 if hasattr(cfg, "scenarios") and cfg.scenarios:
 for scenario in cfg.scenarios:
 if not scenario.get("enable", True):
 vprint(f"--> [SKIPPED] Scenario '{scenario['name']}' is disabled (enable: false).")
 continue
 try:
 perform_single_prediction(cfg, scenario=scenario, abs_results_dir=abs_results_dir)
 vprint(f"--> [SUCCESS] Scenario '{scenario['name']}' completed.")
 except Exception as e:
 import traceback
 vprint(f"--> [ERROR] Scenario '{scenario['name']}' failed: {e}")
 vprint(traceback.format_exc())
 else:
 perform_single_prediction(cfg, abs_results_dir=abs_results_dir)

 vprint("=== All Prediction Tasks Completed ===")


if __name__ == "__main__":
 main()
