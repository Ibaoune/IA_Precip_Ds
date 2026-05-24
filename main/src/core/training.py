"""
==========================================================
 Script: training.py
 Author: M. El Aabaribaoune (@um6p)
 Description:
     Defines the training loop, loss computation, and
     early stopping for the downscaling model.

 Design:
     - Model-agnostic (ViT, CNN, UNet, etc.)
     - GPU / CPU compatible
==========================================================
"""

import torch
import sys
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from src.core.utils import vprint, get_machine_features, estimate_total_time
import time


def _build_model(cfg, x_train, y_train):
    """
    Factory function for model instantiation.
    """
    if cfg.loss_type == "bernoulli_gamma":
        out_channels = 3
    elif cfg.loss_type in ["gaussian", "hurdle_loss"]:
        out_channels = 2
    else:
        out_channels = 1

    if cfg.model_type == "vit":
        from src.models.vit_arch import DownscalingViT
        return DownscalingViT(
            in_channels=x_train.shape[1],
            emb_size=cfg.emb_size,
            patch_size=cfg.patch_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.training_dropout_value if cfg.training_dropout_enable else cfg.dropout,
            output_channels=out_channels,
            n_lat_out=y_train.shape[-2],
            n_lon_out=y_train.shape[-1],
        )
    elif cfg.model_type == "unet":
        from src.models.unet_arch import UNet
        class WrappedUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.unet = UNet(
                    in_channels=x_train.shape[1], 
                    out_channels=out_channels,
                    group_norm_enable=cfg.group_norm_enable,
                    num_groups=cfg.group_norm_num_groups
                )
                self.out_shape = (y_train.shape[-2], y_train.shape[-1])
            def forward(self, x):
                out = self.unet(x)
                if out.shape[-2:] != self.out_shape:
                    out = F.interpolate(out, size=self.out_shape, mode='nearest')
                return out
        return WrappedUNet()
    elif cfg.model_type in ["unet_v2", "unet_coordconv", "attention_unet", "doury_unet"]:
        from src.models.unet_arch import UNet_V2, UNet_CoordConv, Attention_UNet, Doury_UNet
        import torch.nn as nn
        import torch.nn.functional as F
        
        class WrappedUNetAdvanced(nn.Module):
            def __init__(self):
                super().__init__()
                dropout_p = getattr(cfg, "training_dropout_value", getattr(cfg, "dropout", 0.0))
                gn_enable = getattr(cfg, "group_norm_enable", False)
                num_groups = getattr(cfg, "group_norm_num_groups", 32)
                
                if cfg.model_type == "unet_v2":
                    self.unet = UNet_V2(
                        in_channels=x_train.shape[1], out_channels=out_channels,
                        group_norm_enable=gn_enable, num_groups=num_groups, dropout_p=dropout_p
                    )
                elif cfg.model_type == "unet_coordconv":
                    self.unet = UNet_CoordConv(
                        in_channels=x_train.shape[1], out_channels=out_channels,
                        group_norm_enable=gn_enable, num_groups=num_groups, dropout_p=dropout_p
                    )
                elif cfg.model_type == "attention_unet":
                    self.unet = Attention_UNet(
                        in_channels=x_train.shape[1], out_channels=out_channels,
                        group_norm_enable=gn_enable, num_groups=num_groups, dropout_p=dropout_p
                    )
                elif cfg.model_type == "doury_unet":
                    self.unet = Doury_UNet(
                        in_channels=x_train.shape[1], out_channels=out_channels,
                        group_norm_enable=gn_enable, num_groups=num_groups, dropout_p=dropout_p
                    )
                self.out_shape = (y_train.shape[-2], y_train.shape[-1])
            
            def forward(self, x):
                out = self.unet(x)
                if out.shape[-2:] != self.out_shape:
                    out = F.interpolate(out, size=self.out_shape, mode='bilinear', align_corners=True)
                return out
        return WrappedUNetAdvanced()
    elif cfg.model_type == "unet1":
        from src.models.unet_arch1 import UNet as UNet1
        import torch.nn as nn
        import torch.nn.functional as F
        class WrappedUNet1(nn.Module):
            def __init__(self):
                super().__init__()
                self.unet = UNet1(in_channels=x_train.shape[1], out_channels=out_channels)
                self.out_shape = (y_train.shape[-2], y_train.shape[-1])
            def forward(self, x):
                out = self.unet(x)
                if out.shape[-2:] != self.out_shape:
                    out = F.interpolate(out, size=self.out_shape, mode='nearest')
                return out
        return WrappedUNet1()
    elif cfg.model_type == "unet2":
        from src.models.unet_arch2 import UNet as UNet2
        import torch.nn as nn
        import torch.nn.functional as F
        class WrappedUNet2(nn.Module):
            def __init__(self):
                super().__init__()
                self.unet = UNet2(in_channels=x_train.shape[1], out_channels=out_channels)
                self.out_shape = (y_train.shape[-2], y_train.shape[-1])
            def forward(self, x):
                out = self.unet(x)
                if out.shape[-2:] != self.out_shape:
                    out = F.interpolate(out, size=self.out_shape, mode='nearest')
                return out
        return WrappedUNet2()
    elif cfg.model_type == "cnn":
        from src.models.cnn import CNN
        return CNN(
            input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
            out_channels=out_channels,
            output_shape=(y_train.shape[-2], y_train.shape[-1])
        )
    else:
        raise NotImplementedError(f"Model type {cfg.model_type} not supported yet")


def train_model(cfg, x_train, y_train, land_mask=None):
    vprint("Initializing model for training...")

    if cfg.model_type == "glm":
        from src.models.glm import train_glm
        return train_glm(cfg, x_train, y_train, land_mask=land_mask)

    # GPU ADAPTATION: Send the newly built model weights to the target device.
    model = _build_model(cfg, x_train, y_train).to(cfg.device)

    # ----------------
    # Loss & optimizer
    # ----------------
    if cfg.loss_type == "mse":
        criterion = nn.MSELoss()
    elif cfg.loss_type == "asymmetric_mse":
        from src.core.losses import AsymmetricMSELoss
        criterion = AsymmetricMSELoss(alpha=3.0)
    elif cfg.loss_type == "intensity_weighted_mse":
        from src.core.losses import IntensityWeightedMSELoss
        criterion = IntensityWeightedMSELoss(weight_factor=1.0)
    elif cfg.loss_type == "hurdle_loss":
        from src.core.losses import HurdleLoss
        criterion = HurdleLoss(alpha=3.0)
    elif cfg.loss_type == "bernoulli_gamma":
        if cfg.model_type == "vit":
            from src.models.vit_arch import BernoulliGammaLoss
            criterion = BernoulliGammaLoss()
        else:
            from src.core.losses import BernoulliGammaLoss
            criterion = BernoulliGammaLoss()
    elif cfg.loss_type == "gaussian":
        from src.core.losses import GaussianLoss
        criterion = GaussianLoss()
    else:
        raise ValueError(f"Unsupported loss type: {cfg.loss_type}")

    wd_value = cfg.weight_decay_value if cfg.weight_decay_enable else 0.0
    
    if cfg.optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=wd_value)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=wd_value)
    
    plateau_scheduler = None
    if cfg.lr_scheduler_enable:
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=cfg.lr_scheduler_patience, 
            factor=cfg.lr_scheduler_factor, 
            min_lr=cfg.lr_scheduler_min_lr
        )
    
    step_scheduler = None
    if cfg.scheduler_enable:
        if cfg.scheduler_type == "cosine":
            step_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    
    if x_train.shape[0] != y_train.shape[0]:
        print("Mismatch in number of samples!")
        print("x_train samples:", x_train.shape[0])
        print("y_train samples:", y_train.shape[0])
    else:
        print("Same number of samples")
    # ----------------
    # DataLoader
    # ----------------
    full_dataset = TensorDataset(x_train, y_train)
    
    if cfg.validation_enable and cfg.validation_percentage > 0:
        val_size = int(len(full_dataset) * cfg.validation_percentage)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            # GPU ADAPTATION: pin_memory=True speeds up CPU to GPU data transfers (only if data is still on CPU)
            pin_memory=(cfg.device.type == "cuda" and not x_train.is_cuda),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            # GPU ADAPTATION: pin_memory=True speeds up CPU to GPU data transfers (only if data is still on CPU)
            pin_memory=(cfg.device.type == "cuda" and not x_train.is_cuda),
        )
    else:
        train_loader = DataLoader(
            full_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            # GPU ADAPTATION: pin_memory=True speeds up CPU to GPU data transfers (only if data is still on CPU)
            pin_memory=(cfg.device.type == "cuda" and not x_train.is_cuda),
            drop_last=True,
        )
        val_loader = None

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience = 0

    # Training loop
    # ----------------
    for epoch in range(cfg.epochs):
        if epoch == 0:
            features = get_machine_features()
            vprint("\n----------- Machine Features -----------")
            vprint(f" OS: {features['os']}")
            vprint(f" CPU: {features['cpu']} ({features['cores']} cores)")
            vprint(f" RAM: {features['ram_total_gb']} GB")
            if features.get('gpus'):
                for idx, gpu in enumerate(features['gpus']):
                    vprint(f" GPU {idx}: {gpu['name']} ({gpu['memory_total_gb']} GB)")
            else:
                vprint(" GPU: Not available")
            vprint("----------------------------------------")

        model.train()
        total_loss = 0.0
        epoch_start_time = time.time()

        for xb, yb in train_loader:
            # GPU ADAPTATION: Move batch to GPU. non_blocking=True allows overlap of data transfer and compute
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)

            optimizer.zero_grad()
            if total_loss == 0.0:  # Only print for the first batch
                print("DEBUG: xb shape =", xb.shape)
            outputs = model(xb)
            
            if cfg.loss_type == "mse":
                loss_elementwise = (outputs[:, 0, :, :] - yb.squeeze(1)) ** 2
                if land_mask is not None:
                    loss = (loss_elementwise * land_mask).sum() / (land_mask.sum() * outputs.size(0))
                else:
                    loss = loss_elementwise.mean()
            else:
                # BernoulliGammaLoss / GaussianLoss expect (B, C, H, W) and (B, 1, H, W)
                loss = criterion(outputs, yb, mask=land_mask)
                
            loss.backward()

            if cfg.gradient_clipping_enable:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clipping_value)

            optimizer.step()

            total_loss += loss.item()

        epoch_duration = time.time() - epoch_start_time
        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation phase
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    # GPU ADAPTATION: Move validation batch to GPU.
                    xb = xb.to(cfg.device, non_blocking=True)
                    yb = yb.to(cfg.device, non_blocking=True)
                    outputs = model(xb)
                    if cfg.loss_type == "mse":
                        v_loss_elementwise = (outputs[:, 0, :, :] - yb.squeeze(1)) ** 2
                        if land_mask is not None:
                            v_loss = (v_loss_elementwise * land_mask).sum() / (land_mask.sum() * outputs.size(0))
                        else:
                            v_loss = v_loss_elementwise.mean()
                    else:
                        v_loss = criterion(outputs, yb, mask=land_mask)
                    total_val_loss += v_loss.item()
            epoch_val_loss = total_val_loss / len(val_loader)
            val_losses.append(epoch_val_loss)
            vprint(f"Epoch {epoch+1}/{cfg.epochs} - Duration: {epoch_duration:.2f}s - Train Loss: {epoch_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
            monitor_loss = epoch_val_loss
        else:
            vprint(f"Epoch {epoch+1}/{cfg.epochs} - Duration: {epoch_duration:.2f}s - Train Loss: {epoch_loss:.4f}")
            monitor_loss = epoch_loss

        if epoch == 0:
            estimated_time = estimate_total_time(epoch_duration, cfg.epochs)
            vprint(f"Estimated training process duration: {estimated_time}\n")

        if plateau_scheduler is not None:
            plateau_scheduler.step(monitor_loss)
        
        if step_scheduler is not None:
            step_scheduler.step()

        # ----------------
        # Early stopping & Best Weights
        # ----------------
        if monitor_loss < best_val_loss:
            best_val_loss = monitor_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            if cfg.early_stopping_enable:
                patience += 1
                if patience >= cfg.early_stopping_max:
                    vprint("Early stopping triggered.")
                    break

    if best_weights is not None:
        vprint(f"Restoring best weights (Best Loss: {best_val_loss:.4f})")
        model.load_state_dict(best_weights)

    return model, train_losses, val_losses
