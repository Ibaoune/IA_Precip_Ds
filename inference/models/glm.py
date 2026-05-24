# Author: M. El Aabaribaoune (@um6p)
import numpy as np
import statsmodels.api as sm
import os
import pickle
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from src.core.utils import vprint

class PixelWiseGLM:
    """
    Optimized Pixel-Wise GLM for Climate Downscaling.
    
    Features:
    - Bernoulli-Gamma implementation for precipitation.
    - Gaussian implementation for temperature.
    - Matrix-vectorized prediction (orders of magnitude faster than pixel-wise loops).
    - Lightweight storage (only saves model coefficients, not full stat-objects).
    """
    def __init__(self, n_lat, n_lon, n_features, variable="precip"):
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.n_features = n_features
        self.variable = variable
        
        # Buffers for coefficients (params)
        if variable == "precip":
            # [occurrence_params, amount_params]
            self.params_occ = np.full((n_lat, n_lon, n_features), np.nan, dtype=np.float32)
            self.params_amt = np.full((n_lat, n_lon, n_features), np.nan, dtype=np.float32)
        else:
            self.params_gauss = np.full((n_lat, n_lon, n_features), np.nan, dtype=np.float32)

    def predict(self, x):
        """
        Matrix-vectorized prediction.
        x: (N, C, H, W)
        """
        # Interpolate if resolution mismatch
        if x.shape[-2:] != (self.n_lat, self.n_lon):
            import torch.nn.functional as F
            x_torch = torch.from_numpy(x)
            x_torch_interp = F.interpolate(x_torch, size=(self.n_lat, self.n_lon), mode='bilinear', align_corners=False)
            x = x_torch_interp.numpy()

        n_samples, n_channels, n_lat, n_lon = x.shape
        
        # Add constant dimension: (N, C+1, H, W)
        # We append a channel of 1s
        ones = np.ones((n_samples, 1, n_lat, n_lon), dtype=x.dtype)
        X_full = np.concatenate([ones, x], axis=1) # (N, C+1, H, W)

        if self.variable == "precip":
            # Vectorized Matrix Multiplication:
            # Result[n, i, j] = sum_k( X[n, k, i, j] * Params[i, j, k] )
            
            # Rearrange params for einsum or broadcasted mul
            # Params shape: (H, W, C+1) -> (1, C+1, H, W) for broadcasting
            p_occ = self.params_occ.transpose(2, 0, 1)[None, :, :, :]
            p_amt = self.params_amt.transpose(2, 0, 1)[None, :, :, :]
            
            # Link functions
            # 1. Occurrence Probability (Sigmoid for Binomial)
            logits = np.sum(X_full * p_occ, axis=1)
            prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
            
            # 2. Precipitation Amount (Exponential for Gamma log-link)
            log_amount = np.sum(X_full * p_amt, axis=1)
            amount = np.exp(np.clip(log_amount, -10, 10))
            
            predictions = prob * amount
            # Mask pixels with missing models
            mask = np.isnan(self.params_occ[:, :, 0])
            predictions[:, mask] = np.nan
            
        else:
            # Gaussian
            p_gauss = self.params_gauss.transpose(2, 0, 1)[None, :, :, :]
            predictions = np.sum(X_full * p_gauss, axis=1)
            mask = np.isnan(self.params_gauss[:, :, 0])
            predictions[:, mask] = np.nan
            
        return predictions

def _train_pixel(i, j, x_feat, y_point, variable):
    """
    Helper for parallel execution. Fits one pixel.
    """
    if np.isnan(y_point).any():
        return i, j, None

    # Add constant
    X_feat_sm = sm.add_constant(x_feat, has_constant='add')
            
    try:
        if variable == "precip":
            # Binomial for occurrence
            y_binary = (y_point >= 1.0).astype(float)
            if y_binary.sum() < 5: return i, j, None # Not enough rain days
            
            glm_occ = sm.GLM(y_binary, X_feat_sm, family=sm.families.Binomial())
            res_occ = glm_occ.fit()
            
            # Gamma for quantity
            rainy_idx = np.where(y_point >= 1.0)[0]
            if len(rainy_idx) < 15: # Minimum rainy days to fit gamma
                return i, j, None
                
            y_wet = y_point[rainy_idx]
            X_wet = X_feat_sm[rainy_idx]
            glm_amt = sm.GLM(y_wet, X_wet, family=sm.families.Gamma(link=sm.families.links.Log()))
            res_amt = glm_amt.fit()
            
            return i, j, (res_occ.params, res_amt.params)
        else:
            # Gaussian for temperature
            glm_gauss = sm.GLM(y_point, X_feat_sm, family=sm.families.Gaussian())
            res_gauss = glm_gauss.fit()
            return i, j, res_gauss.params
    except:
        return i, j, None

def train_glm(cfg, x_train, y_train, n_jobs=-1):
    """
    Parallelized GLM trainer.
    """
    vprint(f"Starting Parallel GLM training for {cfg.variable} (n_jobs={n_jobs})...")
    
    # Preprocessing
    x_train_np = x_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy().squeeze(1)
    
    if x_train_np.shape[-2:] != y_train_np.shape[-2:]:
        vprint(f"  → Interpolating predictors to {y_train_np.shape[-2:]}")
        import torch.nn.functional as F
        x_torch = torch.from_numpy(x_train_np)
        x_torch_interp = F.interpolate(x_torch, size=y_train_np.shape[-2:], mode='bilinear', align_corners=False)
        x_train_np = x_torch_interp.numpy()

    n_samples, n_channels, n_lat, n_lon = x_train_np.shape
    
    # Initialize wrapper
    glm_wrapper = PixelWiseGLM(n_lat, n_lon, n_channels + 1, variable=cfg.variable)
    
    # Prepare tasks
    tasks = []
    for i in range(n_lat):
        for j in range(n_lon):
            tasks.append((i, j, x_train_np[:, :, i, j], y_train_np[:, i, j], cfg.variable))
    
    # Execute in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_train_pixel)(*t) for t in tqdm(tasks, desc="Parallel Training")
    )
    
    # Collect results
    trained_count = 0
    for i, j, params in results:
        if params is not None:
            if cfg.variable == "precip":
                glm_wrapper.params_occ[i, j] = params[0]
                glm_wrapper.params_amt[i, j] = params[1]
            else:
                glm_wrapper.params_gauss[i, j] = params
            trained_count += 1
                
    vprint(f"GLM training finished. Trained: {trained_count}/{n_lat*n_lon}")
    return glm_wrapper, [], []
