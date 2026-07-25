"""
Author: M. El Aabaribaoune (@um6p)
Description: Utility functions for inference, config generation, and data manipulation.
"""


import os
import sys
import torch
import torch.nn as nn
import numpy as np
import xarray as xr

# Add main directory to sys.path so pickle can resolve 'src' modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../main")))


# Verbose Printing Utility
_VERBOSE = True
def set_verbose(flag: bool):
    global _VERBOSE
    _VERBOSE = flag

def vprint(*args, **kwargs):
    if _VERBOSE:
        print(*args, **kwargs)

# Dataset Utilities
def mask_dataset(ds: xr.Dataset, lon_range: slice, lat_range: slice) -> xr.Dataset:
    """
    Apply spatial mask and print detailed debug information.
    """
    vprint("→ Applying spatial mask")

    lon_name = next(v for v in ds.coords if v.startswith("lon"))
    lat_name = next(v for v in ds.coords if v.startswith("lat"))

    lat_vals = ds[lat_name].values
    if lat_vals[0] > lat_vals[-1]:
        lat_slice = slice(lat_range.stop, lat_range.start)
    else:
        lat_slice = lat_range

    ds_masked = ds.sel(
        **{
            lon_name: lon_range,
            lat_name: lat_slice,
        }
    )

    if ds_masked.sizes[lat_name] == 0 or ds_masked.sizes[lon_name] == 0:
        vprint(">!!< WARNING: EMPTY spatial dimension after mask")

    return ds_masked


# Experiment Path & Model IO
def build_experiment_path(cfg):
    """
    Build a structured experiment path. 
    """
    return cfg.exp_dir


def load_model(cfg, model=None):
    """
    Load trained model checkpoint.
    Supports both torch (.pth) and GLM (.pkl) models.
    """
    if hasattr(cfg, "model_path") and cfg.model_path:
        if os.path.exists(cfg.model_path):
            if cfg.model_path.endswith(".pth"):
                vprint(f" Loading PyTorch model from user-provided path: {cfg.model_path}")
                checkpoint = torch.load(cfg.model_path, map_location="cpu")
                if model is not None:
                    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                return model, checkpoint.get("train_losses"), checkpoint.get("val_losses")
            elif cfg.model_path.endswith(".pkl"):
                vprint(f" Loading GLM model from user-provided path: {cfg.model_path}")
                import pickle
                with open(cfg.model_path, "rb") as f:
                    model = pickle.load(f)
                return model, None, None
            else:
                vprint(f" WARNING: Unknown model extension for {cfg.model_path}. Trying PyTorch load...")
                checkpoint = torch.load(cfg.model_path, map_location="cpu")
                if model is not None:
                    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                return model, checkpoint.get("train_losses"), checkpoint.get("val_losses")
        else:
            vprint(f" WARNING: model_path {cfg.model_path} not found. Falling back to default logic.")

    if hasattr(cfg, "model_save_dir") and cfg.model_save_dir:
        model_dir = cfg.model_save_dir
    else:
        exp_path = build_experiment_path(cfg)
        model_dir = os.path.join(exp_path, "models")
    
    pth_path = os.path.join(model_dir, f"{cfg.model_type}_{cfg.variable}.pth")
    pkl_path = os.path.join(model_dir, f"{cfg.model_type}_{cfg.variable}.pkl")

    if os.path.exists(pth_path):
        vprint(f" Loading PyTorch model from: {pth_path}")
        checkpoint = torch.load(pth_path, map_location="cpu")
        if model is not None:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        return model, checkpoint.get("train_losses"), checkpoint.get("val_losses")
    
    elif os.path.exists(pkl_path):
        vprint(f" Loading GLM model from: {pkl_path}")
        import pickle
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        return model, None, None
    
    else:
        raise FileNotFoundError(f"No model found for {cfg.model_type}_{cfg.variable} in {model_dir}")


def _build_model(cfg, x_test, y_test):
    """
    Constructs the model architecture based on the configuration.
    """
    if cfg.loss_type == "bernoulli_gamma":
        out_channels = 3
    elif cfg.loss_type == "gaussian":
        out_channels = 2
    else:
        out_channels = 1

    if cfg.model_type == "vit":
        from models.vit_arch import DownscalingViT
        return DownscalingViT(
            in_channels=x_test.shape[1],
            emb_size=cfg.emb_size,
            patch_size=cfg.patch_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            output_channels=out_channels,
            n_lat_out=y_test.shape[-2],
            n_lon_out=y_test.shape[-1],
        )
    elif cfg.model_type == "unet":
        from models.unet_arch import UNet
        import torch.nn as nn
        import torch.nn.functional as F
        class WrappedUNet(nn.Module):
            def __init__(self):
                super().__init__()
                out_channels = 3 if cfg.loss_type == "bernoulli_gamma" else 1
                self.unet = UNet(
                    in_channels=x_test.shape[1], 
                    out_channels=out_channels,
                    group_norm_enable=cfg.group_norm_enable,
                    num_groups=cfg.group_norm_num_groups
                )
                self.out_shape = (y_test.shape[-2], y_test.shape[-1])
            def forward(self, x):
                out = self.unet(x)
                if out.shape[-2:] != self.out_shape:
                    out = F.interpolate(out, size=self.out_shape, mode='nearest')
                return out
        return WrappedUNet()
    elif cfg.model_type == "unet1":
        from models.unet_arch1 import UNet as UNet1
        import torch.nn as nn
        import torch.nn.functional as F
        class WrappedUNet1(nn.Module):
            def __init__(self):
                super().__init__()
                out_channels = 3 if cfg.loss_type == "bernoulli_gamma" else 1
                self.unet = UNet1(
                    in_channels=x_test.shape[1], 
                    out_channels=out_channels,
                    group_norm_enable=cfg.group_norm_enable,
                    num_groups=cfg.group_norm_num_groups
                )
                self.out_shape = (y_test.shape[-2], y_test.shape[-1])
            def forward(self, x):
                out = self.unet(x)
                if out.shape[-2:] != self.out_shape:
                    out = F.interpolate(out, size=self.out_shape, mode='nearest')
                return out
        return WrappedUNet1()
    elif cfg.model_type == "unet2":
        from models.unet_arch2 import UNet as UNet2
        import torch.nn as nn
        import torch.nn.functional as F
        class WrappedUNet2(nn.Module):
            def __init__(self):
                super().__init__()
                out_channels = 3 if cfg.loss_type == "bernoulli_gamma" else 1
                self.unet = UNet2(
                    in_channels=x_test.shape[1], 
                    out_channels=out_channels,
                    group_norm_enable=cfg.group_norm_enable,
                    num_groups=cfg.group_norm_num_groups
                )
                self.out_shape = (y_test.shape[-2], y_test.shape[-1])
            def forward(self, x):
                out = self.unet(x)
                if out.shape[-2:] != self.out_shape:
                    out = F.interpolate(out, size=self.out_shape, mode='nearest')
                return out
        return WrappedUNet2()
    elif cfg.model_type == "cnn":
        from models.cnn import CNN
        out_channels = 3 if cfg.loss_type == "bernoulli_gamma" else 1
        return CNN(
            input_shape=(x_test.shape[1], x_test.shape[2], x_test.shape[3]),
            out_channels=out_channels,
            output_shape=(y_test.shape[-2], y_test.shape[-1])
        )
    else:
        raise NotImplementedError(f"Model {cfg.model_type} not supported")
