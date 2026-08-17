# =============================================================================
# RETAINED U-NET ARCHITECTURE — unet_exp32_parallel
# =============================================================================
# Author : M. El Aabaribaoune (@um6p)
# Date   : 2026-08-12
# Status : PRODUCTION — This is the officially retained U-Net architecture
#          for the downscaling paper. Do NOT modify without updating
#          docs/unet_troubleshooting_report.md.
#
# Architecture  : Parallel CNN + U-Net Hybrid (exp32)
# Training cfg  : main/results/unet/tests/tests_advanced_arch/unet_exp32_parallel/
# Weights       : .../gridbox_0.0001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn/
#                 models/unet_exp32_precip.pth
# Inference cfg : inference/tests/config_unet_exp32_parallel_1979_2014.yaml
# LMDZ output   : inference/results/1979_2014/unet_exp32_parallel/
#                 unet_exp32_lmdz_250_present_true.nc
#
# Design
# ------
# The model has two parallel branches that run on the same CoordConv-augmented
# predictor tensor:
#
#   Branch 1 — CNN_Exp5 (BaselineCNNBranch)
#     Conv50 → Conv25 → Conv1 → Flatten → Dense(fc_input → output_pixels)
#     Role: captures the smooth, domain-wide mean precipitation signal.
#
#   Branch 2 — Shallow U-Net (Encoder-Decoder with skip connections)
#     Enc[in→50] → MaxPool → Enc[50→25] → MaxPool → Bot[25→25]
#     → Upsample → Dec[50→50] → Upsample → Dec[100→25] → Conv1×1
#     Role: recovers local extremes, orographic gradients, and residuals.
#
#   Output = CNN_branch + UNet_branch   (element-wise sum)
#
# Loss      : Bernoulli-Gamma (occurrence + log-shape + log-scale)
# Norm mode : gridbox   (pixel-wise ERA5 standardization)
#
# Key lessons (see docs/unet_troubleshooting_report.md §8–9)
# ----------------------------------------------------------
# * Earlier LMDZ collapse (flat 0.5 mm/day) was a PyTorch state-dict loading
#   bug (missing WrappedUNetExp wrapper), NOT an architectural failure.
# * After the bug fix, exp32 transfers correctly to LMDZ and outperforms
#   CNN_Exp5 in all four Moroccan regions (1979–2014).
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet_arch import AddCoords


# ---------------------------------------------------------------------------
# Sub-module: CNN_Exp5 branch (replicates BaselineCNNBranch exactly)
# ---------------------------------------------------------------------------
class _CNNBranch(nn.Module):
    """
    Exact replica of CNN_Exp5 (BaselineCNNBranch).
    3-layer conv stack → flatten → one Dense FC layer per output channel.
    Captures domain-wide mean precipitation signal.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 output_shape: tuple, input_spatial_shape: tuple = (9, 10)):
        super().__init__()
        self.output_shape = output_shape
        self.out_channels  = out_channels

        self.conv1 = nn.Conv2d(in_channels, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50,          25, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(25,           1, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()

        fc_in = 1 * input_spatial_shape[0] * input_spatial_shape[1]
        self.fc_layers = nn.ModuleList([
            nn.Linear(fc_in, output_shape[0] * output_shape[1])
            for _ in range(out_channels)
        ])

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        params = [fc(x).view(B, 1, *self.output_shape) for fc in self.fc_layers]
        return torch.cat(params, dim=1)


# ---------------------------------------------------------------------------
# Retained architecture: UNet_Exp32_Parallel
# ---------------------------------------------------------------------------
class UNet_Exp32_Parallel(nn.Module):
    """
    Retained U-Net architecture for precipitation downscaling (exp32_parallel).

    Two parallel branches share the same CoordConv-augmented input:
      - CNN branch  : global mean/bias correction (CNN_Exp5 replica)
      - U-Net branch: local extremes and residuals (shallow 2-level encoder-decoder)

    The final prediction is the element-wise sum of both branches.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 output_shape: tuple, input_spatial_shape: tuple = (9, 10)):
        """
        Parameters
        ----------
        in_channels         : number of atmospheric predictor channels
        out_channels        : number of Bernoulli-Gamma output channels (3)
        output_shape        : (H, W) of the 10-km target grid, e.g. (160, 180)
        input_spatial_shape : (H, W) of the coarse predictor grid, e.g. (9, 10)
        """
        super().__init__()
        self.output_shape = output_shape
        self.add_coords   = AddCoords()
        in_c = in_channels + 2   # +2 for injected lat/lon coordinate channels

        # ── Branch 1: CNN_Exp5 (global mean) ──────────────────────────────
        self.cnn_branch = _CNNBranch(in_c, out_channels, output_shape, input_spatial_shape)

        # ── Branch 2: Shallow U-Net (local residuals) ─────────────────────
        # Encoder
        self.enc1  = nn.Sequential(nn.Conv2d(in_c, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2  = nn.Sequential(nn.Conv2d(50,  25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bot   = nn.Sequential(nn.Conv2d(25, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        # Decoder (with skip connections)
        self.up1   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1  = nn.Sequential(nn.Conv2d(50,  50, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        self.up2   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2  = nn.Sequential(nn.Conv2d(100, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        self.final_unet = nn.Conv2d(25, out_channels, kernel_size=1)

    # -----------------------------------------------------------------------
    def _match_size(self, x, shape):
        """Bilinear resize to (H, W) if sizes differ."""
        if x.shape[-2:] != shape:
            return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x

    # -----------------------------------------------------------------------
    def forward(self, x):
        x = self.add_coords(x)            # inject (lat, lon) coordinates

        # ── Branch 1 ──────────────────────────────────────────────────────
        cnn_out = self.cnn_branch(x)       # (B, out_channels, H_out, W_out)

        # ── Branch 2 ──────────────────────────────────────────────────────
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bot(self.pool2(e2))

        d1 = self.up1(b)
        d1 = self._match_size(d1, e2.shape[-2:])
        d1 = self.dec1(torch.cat([d1, e2], dim=1))

        d2 = self.up2(d1)
        d2 = self._match_size(d2, e1.shape[-2:])
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        unet_out = self.final_unet(d2)
        unet_out = self._match_size(unet_out, self.output_shape)

        # ── Combine ───────────────────────────────────────────────────────
        return cnn_out + unet_out
