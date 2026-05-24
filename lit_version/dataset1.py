import os
import logging
import torch
import numpy as np
import xarray as xr
import pandas as pd
from torch.utils.data import Dataset


# ============================================================
# LOGGER
# ============================================================

def _setup_logger(name: str, verbose: int = 1):
    logger = logging.getLogger(name)
    logger.propagate = False

    if verbose == 0:
        level = logging.WARNING
    elif verbose == 2:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s] %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ============================================================
# DATASET
# ============================================================

class DownscalingDataset(Dataset):

    def __init__(
        self,
        mode="train",
        variables=("z", "q", "u", "v", "t"),
        levels=("1000", "850", "700", "500"),
        input_normalize=None,
        target_normalize=None,
        return_date=False,
        extent=(-10, 0, 29, 36),
        data_path=None,
        start_date=None,
        end_date=None,
        stats_path="data_stats/",
        input_ds_name="era5",
        target_ds_name="mswp",
        log_transform=False,
        verbose=None,
    ):

        # ---- verbosity ----
        if verbose is None:
            self.verbose = 1
        elif isinstance(verbose, bool):
            self.verbose = 2 if verbose else 0
        else:
            self.verbose = int(verbose)

        self.logger = _setup_logger(self.__class__.__name__, self.verbose)
        self.logger.info(f"[{mode.upper()} DATASET INITIALIZATION]")

        # ---- basic attributes ----
        self.mode = mode
        self.variables = variables
        self.levels = levels
        self.extent = extent
        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.stats_path = stats_path
        self.input_normalize = input_normalize
        self.target_normalize = target_normalize
        self.log_transform = log_transform
        self.return_date = return_date
        self.input_ds_name = input_ds_name.lower()
        self.target_ds_name = target_ds_name.lower()

        os.makedirs(self.stats_path, exist_ok=True)

        # ---- target units ----
        self.target_unit = self._infer_target_unit()

        # ---- load data ----
        self.logger.info("Preparing input dataset")
        self.x_data = self._prepare_inputs()

        self.logger.info("Preparing target dataset")
        self.y_data = self._prepare_targets()

        assert self.x_data.sizes["time"] == self.y_data.sizes["time"], \
            "Input and target time dimensions do not match"

        # ---- statistics ----
        self.logger.info("Computing / loading input statistics")
        self.x_mean, self.x_std = self._compute_stats(
            self.x_data, "x", self.input_normalize, self.input_ds_name
        )

        self.logger.info("Computing / loading target statistics")
        self.y_mean, self.y_std = self._compute_stats(
            self.y_data, "y", self.target_normalize, self.target_ds_name
        )

        self.n_channels = len(self.variables) * len(self.levels)
        self.output_shape = (len(self.lat), len(self.lon))

        self.logger.debug(f"Dataset ready: {len(self)} samples")
        self.logger.debug(f"Input channels: {self.n_channels}")
        self.logger.debug(f"Output shape: {self.output_shape}")

    # ============================================================
    # IO
    # ============================================================

    def _open_input_dataset(self):
        self.logger.debug("Opening input files")
        return xr.open_mfdataset(
            self.data_path["input"],
            combine="by_coords",
            preprocess=rename_valid_time,
        )

    def _open_target_dataset(self):
        self.logger.debug("Opening target files")
        ds = xr.open_mfdataset(
            self.data_path["target"],
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine="nested",
            concat_dim="time",
            preprocess=rename_valid_time,
        )
        var = list(ds.data_vars.keys())[0]
        return ds, var

    # ============================================================
    # COMMON PREPROCESSING
    # ============================================================

    def _standardize_and_slice(self, ds):
        self.logger.debug("Standardizing coordinates")

        ds = ds.rename(
            {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items()
             if k in ds.dims}
        )

        ds = ds.sortby("time")

        if ds.lat[0] < ds.lat[-1]:
            self.logger.debug("Reversing latitude order")
            ds = ds.reindex(lat=list(reversed(ds.lat)))

        lon = ds.lon.values
        if not np.any(lon < 0) or np.any(lon > 180):
            self.logger.debug("Wrapping longitude to [-180, 180]")
            ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)
            ds = ds.sortby("lon")

        self.logger.debug(f"Slicing spatial extent: {self.extent}")
        ds = ds.sel(
            lon=slice(self.extent[0], self.extent[1]),
            lat=slice(self.extent[3], self.extent[2]),
        )

        self.logger.debug("Slicing temporal extent")
        ds = ds.sel(time=slice(self.start_date, self.end_date))

        self.logger.debug(f"Dataset dims after slice: {ds.dims}")
        return ds

    def _ensure_time_frequency(self, ds, name, method):
        freq = None
        try:
            freq = xr.infer_freq(ds.time.to_index())
        except Exception:
            pass

        self.logger.debug(f"{name} inferred frequency: {freq}")

        if freq == "D":
            return ds

        if freq and freq.endswith("h"):
            self.logger.info(f"Resampling {name} {freq} → daily ({method})")
            return getattr(ds.resample(time="1D"), method)()

        self.logger.warning(
            f"Could not infer {name} frequency (freq={freq}), assuming daily"
        )
        return ds

    # ============================================================
    # INPUT PIPELINE
    # ============================================================

    def _prepare_inputs(self):
        ds = self._open_input_dataset()
        ds = self._standardize_and_slice(ds)
        ds = ds.sel(level=self.levels)[self.variables]

        ds = self._ensure_time_frequency(ds, "input", "mean")

        self.logger.debug("Loading input data into memory")
        return ds.load()

    # ============================================================
    # TARGET PIPELINE
    # ============================================================

    def _prepare_targets(self):
        ds, var = self._open_target_dataset()
        ds = self._standardize_and_slice(ds)
        ds = self._ensure_time_frequency(ds, "target", "sum")

        y = ds[var]

        self.logger.debug("Converting target units")
        y = self._convert_target_units(y)

        if self.log_transform:
            self.logger.info("Applying log1p transform to target")
            y = np.log1p(y)

        self._set_target_coords(y)

        self.logger.debug("Loading target data into memory")
        return y.load()

    def _set_target_coords(self, y):
        self.time = y.time.values
        self.lat = y.lat.values
        self.lon = y.lon.values

    # ============================================================
    # UNITS / STATS
    # ============================================================

    def _infer_target_unit(self):
        if self.target_ds_name in ["lmdz", "era5"]:
            return "kg/m2/s"
        if self.target_ds_name in ["mswp", "ter"]:
            return "mm/day"
        if self.target_ds_name == "imerg":
            return "mm/hr"
        raise ValueError(f"Unknown target dataset: {self.target_ds_name}")

    def _convert_target_units(self, y):
        if self.target_unit in ["kg/m2/s", "kg/m^2/s"]:
            self.logger.debug("kg/m2/s → mm/day")
            return y * 86400.0
        if self.target_unit in ["mm/hr"]:
            self.logger.debug("mm/hr → mm/day")
            return y * 24.0
        return y

    def _compute_stats(self, ds, data_type, normalization, ds_name):
        if normalization is None or normalization == "per_day":
            self.logger.debug(f"No stats needed for {data_type}")
            return None, None

        suffix = "_log" if self.log_transform and data_type == "y" else ""
        stats_file = os.path.join(
            self.stats_path,
            f"{data_type}_stats_{normalization}_{ds_name}{suffix}.npz"
        )

        if self.mode == "train":
            self.logger.debug(f"Computing {data_type} stats ({normalization})")

            if normalization == "per_channel":
                mean = np.concatenate(
                    [ds[v].mean(("time", "lat", "lon")).values for v in ds.data_vars]
                )[:, None, None]
                std = np.concatenate(
                    [ds[v].std(("time", "lat", "lon")).values for v in ds.data_vars]
                )[:, None, None]
            elif normalization == "global":
                mean = ds.mean().values
                std = ds.std().values
            else:
                raise ValueError(normalization)

            np.savez(stats_file, mean=mean, std=std)
            self.logger.info(f"Saved stats → {stats_file}")
            return mean, std

        self.logger.info(f"Loading stats ← {stats_file}")
        stats = np.load(stats_file)
        return stats["mean"], stats["std"]

    # ============================================================
    # PYTORCH API
    # ============================================================

    def __len__(self):
        return self.x_data.sizes["time"]

    def __getitem__(self, idx):
        date = self.time[idx]
        dayofyear = pd.to_datetime(date).dayofyear

        x_ds = self.x_data.isel(time=idx)
        x = np.concatenate([x_ds[v].values for v in x_ds.data_vars], axis=0)
        x = torch.from_numpy(x)

        y = torch.from_numpy(self.y_data.isel(time=idx).values)

        if self.input_normalize:
            x = (x - self.x_mean) / (self.x_std + 1e-8)

        if self.target_normalize:
            y = (y - self.y_mean) / (self.y_std + 1e-8)

        seasonal = torch.tensor([
            np.cos(2 * np.pi * dayofyear / 365),
            np.sin(2 * np.pi * dayofyear / 365)
        ], dtype=torch.float32)

        if self.return_date:
            return x.float(), y.float(), seasonal, str(date)

        return x.float(), y.float(), seasonal


# ============================================================
# HELPERS
# ============================================================

def rename_valid_time(ds):
    if "valid_time" in ds.dims:
        return ds.rename({"valid_time": "time"})
    if "counter_time" in ds.dims:
        return ds.rename({"counter_time": "time"})
    return ds
