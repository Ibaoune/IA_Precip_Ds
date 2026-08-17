import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.unet_arch import ConvBlock, UpSampleBlock, AddCoords, UNet_V2

# ---------------------------------------------------------
# Config 1: FCN Simple (CNN without Dense, using Upsample)
# ---------------------------------------------------------
class UNet_Config1(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.conv1 = nn.Conv2d(in_channels, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(25, 16, kernel_size=3, padding=1)
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=True)
        return self.final(x)

# ---------------------------------------------------------
# Config 2: FCN + CoordConv
# ---------------------------------------------------------
class UNet_Config2(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.add_coords = AddCoords()
        self.conv1 = nn.Conv2d(in_channels + 2, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(25, 16, kernel_size=3, padding=1)
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.add_coords(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=True)
        return self.final(x)

# ---------------------------------------------------------
# Config 3: FCN bottleneck (AutoEncoder sans skip)
# ---------------------------------------------------------
class UNet_Config3(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.enc1 = ConvBlock(in_channels, 32)
        self.pool = nn.MaxPool2d(2, 2)
        self.bot = ConvBlock(32, 64)
        self.up = UpSampleBlock(64, 32)
        self.dec1 = ConvBlock(32, 32)
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        bot = self.bot(self.pool(e1))
        d1 = self.up(bot)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(d1)
        d1 = F.interpolate(d1, size=self.output_shape, mode='bilinear', align_corners=True)
        return self.final(d1)

# ---------------------------------------------------------
# Config 4: AutoEncoder + 1 Global Skip (Residual)
# ---------------------------------------------------------
class UNet_Config4(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.enc1 = ConvBlock(in_channels, 32)
        self.pool = nn.MaxPool2d(2, 2)
        self.bot = ConvBlock(32, 64)
        self.up = UpSampleBlock(64, 32)
        self.dec1 = ConvBlock(32, 32)
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        # Residual mapping from input
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x_in):
        e1 = self.enc1(x_in)
        bot = self.bot(self.pool(e1))
        d1 = self.up(bot)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(d1)
        out = self.final(d1)
        out = F.interpolate(out, size=self.output_shape, mode='bilinear', align_corners=True)
        
        # Global Skip Connection (Residual)
        res = self.skip_conv(x_in)
        res = F.interpolate(res, size=self.output_shape, mode='bilinear', align_corners=True)
        return out + res

# ---------------------------------------------------------
# Config 5: Shallow U-Net (1 level + Concatenation Skip)
# ---------------------------------------------------------
class UNet_Config5(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.bot = ConvBlock(64, 128)
        self.up = UpSampleBlock(128, 64)
        # In channels = 64 (from up) + 64 (from skip) = 128
        self.dec1 = ConvBlock(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        bot = self.bot(self.pool(e1))
        up = self.up(bot)
        if up.shape[-2:] != e1.shape[-2:]:
            up = F.interpolate(up, size=e1.shape[-2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([up, e1], dim=1))
        d1 = F.interpolate(d1, size=self.output_shape, mode='bilinear', align_corners=True)
        return self.final(d1)

# ---------------------------------------------------------
# Config 6: Shallow U-Net + CoordConv
# ---------------------------------------------------------
class UNet_Config6(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.add_coords = AddCoords()
        self.enc1 = ConvBlock(in_channels + 2, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.bot = ConvBlock(64, 128)
        self.up = UpSampleBlock(128, 64)
        self.dec1 = ConvBlock(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.add_coords(x)
        e1 = self.enc1(x)
        bot = self.bot(self.pool(e1))
        up = self.up(bot)
        if up.shape[-2:] != e1.shape[-2:]:
            up = F.interpolate(up, size=e1.shape[-2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([up, e1], dim=1))
        d1 = F.interpolate(d1, size=self.output_shape, mode='bilinear', align_corners=True)
        return self.final(d1)

# ---------------------------------------------------------
# Config 7: U-Net Standard (2 levels)
# ---------------------------------------------------------
class UNet_Config7(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.bot = ConvBlock(128, 256)

        self.up2 = UpSampleBlock(256, 128)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = UpSampleBlock(128, 64)
        self.dec1 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        bot = self.bot(self.pool(e2))

        up2 = self.up2(bot)
        if up2.shape[-2:] != e2.shape[-2:]:
            up2 = F.interpolate(up2, size=e2.shape[-2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))

        up1 = self.up1(d2)
        if up1.shape[-2:] != e1.shape[-2:]:
            up1 = F.interpolate(up1, size=e1.shape[-2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))
        
        d1 = F.interpolate(d1, size=self.output_shape, mode='bilinear', align_corners=True)
        return self.final(d1)

# ---------------------------------------------------------
# Config 8: U-Net Standard (3 levels) - Equivalent to UNet_V2
# ---------------------------------------------------------
class UNet_Config8(UNet_V2):
    def __init__(self, in_channels, out_channels, output_shape):
        super().__init__(in_channels=in_channels, out_channels=out_channels, 
                         group_norm_enable=False, dropout_p=0.0)
        self.output_shape_final = output_shape
    def forward(self, x):
        out = super().forward(x)
        if out.shape[-2:] != self.output_shape_final:
            out = F.interpolate(out, size=self.output_shape_final, mode='bilinear', align_corners=True)
        return out

# ---------------------------------------------------------
# Config 9: UNet V2 + CoordConv
# ---------------------------------------------------------
from src.models.unet_arch import UNet_CoordConv
class UNet_Config9(UNet_CoordConv):
    def __init__(self, in_channels, out_channels, output_shape):
        super().__init__(in_channels=in_channels, out_channels=out_channels, 
                         group_norm_enable=False, dropout_p=0.0)
        self.output_shape_final = output_shape
    def forward(self, x):
        out = super().forward(x)
        if out.shape[-2:] != self.output_shape_final:
            out = F.interpolate(out, size=self.output_shape_final, mode='bilinear', align_corners=True)
        return out

# ---------------------------------------------------------
# Config 10: U-Net Hybride Dense (CNN-UNet)
# ---------------------------------------------------------
import numpy as np
class UNet_Config10(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.out_channels = out_channels
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.bot = ConvBlock(128, 256)

        self.up2 = UpSampleBlock(256, 128)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = UpSampleBlock(128, 64)
        self.dec1 = ConvBlock(128, 64)
        
        # Adaptive pooling to fix spatial dimension before linear
        self.adaptive_pool = nn.AdaptiveAvgPool2d((10, 10))
        self.flatten = nn.Flatten()
        
        # A dense layer per output channel, similar to CNN
        self.fc_layers = nn.ModuleList([
            nn.Linear(64 * 10 * 10, np.prod(output_shape)) 
            for _ in range(out_channels)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        bot = self.bot(self.pool(e2))

        up2 = self.up2(bot)
        if up2.shape[-2:] != e2.shape[-2:]:
            up2 = F.interpolate(up2, size=e2.shape[-2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))

        up1 = self.up1(d2)
        if up1.shape[-2:] != e1.shape[-2:]:
            up1 = F.interpolate(up1, size=e1.shape[-2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))
        
        # Instead of Conv2d, we use the Dense projection
        d1_pooled = self.adaptive_pool(d1)
        d1_flat = self.flatten(d1_pooled)
        
        params = []
        for fc in self.fc_layers:
            p = fc(d1_flat)
            params.append(p.view(batch_size, 1, self.output_shape[0], self.output_shape[1]))

        outputs = torch.cat(params, dim=1)
        return outputs


# ---------------------------------------------------------
# Hybrid Helper: Baseline CNN Branch
# ---------------------------------------------------------
class BaselineCNNBranch(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape, input_spatial_shape=(9, 10)):
        super().__init__()
        self.output_shape = output_shape
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(25, 1, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        
        fc_input_size = 1 * input_spatial_shape[0] * input_spatial_shape[1]
        
        self.fc_layers = nn.ModuleList([
            nn.Linear(fc_input_size, output_shape[0] * output_shape[1]) 
            for _ in range(out_channels)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        
        params = []
        for fc in self.fc_layers:
            p = fc(x)
            params.append(p.view(batch_size, 1, self.output_shape[0], self.output_shape[1]))
            
        return torch.cat(params, dim=1)

# ---------------------------------------------------------
# Config 23: Parallel CNN + UNet (The Hybrid)
# ---------------------------------------------------------
class UNet_Config23(nn.Module):
    """
    Two branches: one exact CNN for global mean bias, one standard U-Net for spatial extremes.
    """
    def __init__(self, in_channels, out_channels, output_shape, input_spatial_shape=(9, 10)):
        super().__init__()
        self.add_coords = AddCoords()
        in_c = in_channels + 2
        
        # Branch 1: CNN
        self.cnn_branch = BaselineCNNBranch(in_c, out_channels, output_shape, input_spatial_shape)
        
        # Branch 2: Standard U-Net (similar to Config 17)
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.bot = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(100, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(nn.Conv2d(75, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.final_unet = nn.Conv2d(25, out_channels, kernel_size=1)
        self.output_shape = output_shape
        
    def _match_size(self, x, shape):
        if x.shape[-2:] != shape:
            return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        x = self.add_coords(x)
        
        # Branch 1
        cnn_out = self.cnn_branch(x)
        
        # Branch 2
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))
        
        d1 = self.up1(b)
        d1 = self._match_size(d1, e2.shape[-2:])
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        
        d2 = self.up2(d1)
        d2 = self._match_size(d2, e1.shape[-2:])
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        unet_out = self.final_unet(d2)
        unet_out = self._match_size(unet_out, self.output_shape)
        
        # Sum both outputs
        return cnn_out + unet_out

# ---------------------------------------------------------
# Config 24: UNet with FC Bottleneck
# ---------------------------------------------------------
class UNet_Config24(nn.Module):
    """
    A U-Net where the bottleneck passes through a Linear layer, 
    allowing for a global shift to be learned at the lowest resolution.
    """
    def __init__(self, in_channels, out_channels, output_shape, input_spatial_shape=(9, 10)):
        super().__init__()
        self.add_coords = AddCoords()
        in_c = in_channels + 2
        self.output_shape = output_shape
        
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.bot_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        # Calculate bottleneck size
        h, w = input_spatial_shape
        h, w = h // 4, w // 4
        self.bot_h, self.bot_w = h, w
        self.flatten = nn.Flatten()
        
        self.bot_fc = nn.Sequential(
            nn.Linear(64 * h * w, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128 * h * w),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def _match_size(self, x, shape):
        if x.shape[-2:] != shape:
            return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        x = self.add_coords(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        b = F.relu(self.bot_conv1(self.pool2(e2)))
        b_flat = self.flatten(b)
        b_fc = self.bot_fc(b_flat)
        b_reshaped = b_fc.view(x.size(0), 128, self.bot_h, self.bot_w)
        
        d1 = self.up1(b_reshaped)
        d1 = self._match_size(d1, e2.shape[-2:])
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        
        d2 = self.up2(d1)
        d2 = self._match_size(d2, e1.shape[-2:])
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        out = self.final(d2)
        return self._match_size(out, self.output_shape)

# ---------------------------------------------------------
# Config 25: Global Average Pooling Bias Corrector
# ---------------------------------------------------------
class UNet_Config25(nn.Module):
    """
    Standard U-Net but adds a parallel branch that applies GAP + MLP to output a global spatial bias map.
    """
    def __init__(self, in_channels, out_channels, output_shape):
        super().__init__()
        self.add_coords = AddCoords()
        in_c = in_channels + 2
        self.output_shape = output_shape
        
        # Standard UNet
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.bot = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # GAP Bias Corrector
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_bias = nn.Sequential(
            nn.Linear(in_c, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_channels * output_shape[0] * output_shape[1])
        )

    def _match_size(self, x, shape):
        if x.shape[-2:] != shape:
            return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        x = self.add_coords(x)
        
        # U-Net path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))
        
        d1 = self.up1(b)
        d1 = self._match_size(d1, e2.shape[-2:])
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        
        d2 = self.up2(d1)
        d2 = self._match_size(d2, e1.shape[-2:])
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        unet_out = self.final(d2)
        unet_out = self._match_size(unet_out, self.output_shape)
        
        # Bias path
        gap_feat = self.gap(x).view(x.size(0), -1)
        bias_out = self.fc_bias(gap_feat)
        bias_out = bias_out.view(x.size(0), -1, self.output_shape[0], self.output_shape[1])
        
        return unet_out + bias_out

# ---------------------------------------------------------
# Config 26: CNN-Guided UNet Decoder
# ---------------------------------------------------------
class UNet_Config26(nn.Module):
    """
    Uses the exact CNN 3-layer architecture as the encoder (no pooling), 
    and a decoder to project it directly to the output resolution.
    """
    def __init__(self, in_channels, out_channels, output_shape):
        super().__init__()
        self.add_coords = AddCoords()
        in_c = in_channels + 2
        self.output_shape = output_shape
        
        # Encoder (CNN structure)
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(50, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(25, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.bot = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(nn.Conv2d(75, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(100, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.final = nn.Conv2d(25, out_channels, kernel_size=1)
        
    def _match_size(self, x, shape):
        if x.shape[-2:] != shape:
            return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        x = self.add_coords(x)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        b = self.bot(e3)
        b = self._match_size(b, self.output_shape)
        e2_proj = self._match_size(e2, self.output_shape)
        e1_proj = self._match_size(e1, self.output_shape)
        
        d1 = self.dec1(torch.cat([b, e2_proj], dim=1))
        d2 = self.dec2(torch.cat([d1, e1_proj], dim=1))
        
        out = self.final(d2)
        return out

# ---------------------------------------------------------
# Config 27: Deep Parallel Hybrid (CNN-Dense + UNet-Dilated)
# ---------------------------------------------------------
class UNet_Config27(nn.Module):
    """
    Combines Dilated UNet (Config 18) and CNN with a learnable gate.
    """
    def __init__(self, in_channels, out_channels, output_shape, input_spatial_shape=(9, 10)):
        super().__init__()
        self.add_coords = AddCoords()
        in_c = in_channels + 2
        self.output_shape = output_shape
        
        # Branch 1: CNN
        self.cnn_branch = BaselineCNNBranch(in_c, out_channels, output_shape, input_spatial_shape)
        
        # Branch 2: Dilated U-Net
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, 64, kernel_size=3, padding=1), nn.ELU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ELU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.bot = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2, dilation=2), nn.ELU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2), nn.ELU(inplace=True)
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.ELU(inplace=True))
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1), nn.ELU(inplace=True))
        
        self.final_unet = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # Learnable gating for combining them
        self.gate = nn.Parameter(torch.tensor([0.5]))

    def _match_size(self, x, shape):
        if x.shape[-2:] != shape:
            return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        x = self.add_coords(x)
        
        cnn_out = self.cnn_branch(x)
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))
        
        d1 = self.up1(b)
        d1 = self._match_size(d1, e2.shape[-2:])
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        
        d2 = self.up2(d1)
        d2 = self._match_size(d2, e1.shape[-2:])
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        unet_out = self.final_unet(d2)
        unet_out = self._match_size(unet_out, self.output_shape)
        
        gate = torch.sigmoid(self.gate)
        return gate * cnn_out + (1 - gate) * unet_out


# =========================================================
# EXPERIMENTS 28-31: CNN_Exp5 BASED U-NET ARCHITECTURES
# =========================================================

# ---------------------------------------------------------
# Config 28: Variant 1 (Strict CNN_Exp5 Base U-Net)
# ---------------------------------------------------------
class UNet_Config28(nn.Module):
    """
    Variant 1: Strict CNN_Exp5 Base U-Net
    """
    def __init__(self, in_channels, out_channels, output_shape, input_spatial_shape=(9, 10)):
        super().__init__()
        self.output_shape = output_shape
        self.add_coords = AddCoords()
        in_c = in_channels + 2
        
        # Encoder (50 -> 25)
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(50, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bot = nn.Sequential(nn.Conv2d(25, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(nn.Conv2d(100, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.final = nn.Conv2d(25, out_channels, kernel_size=1)
        
    def _match_size(self, x, shape):
        if x.shape[-2:] != shape:
            return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        x = self.add_coords(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))
        
        d1 = self.up1(b)
        d1 = self._match_size(d1, e2.shape[-2:])
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        
        d2 = self.up2(d1)
        d2 = self._match_size(d2, e1.shape[-2:])
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        out = self.final(d2)
        return self._match_size(out, self.output_shape)

# ---------------------------------------------------------
# Config 29: Variant 2 (Dense Bottleneck CNN_Exp5 U-Net)
# ---------------------------------------------------------
class UNet_Config29(nn.Module):
    """
    Variant 2: Dense Bottleneck CNN_Exp5 U-Net
    """
    def __init__(self, in_channels, out_channels, output_shape, input_spatial_shape=(9, 10)):
        super().__init__()
        self.output_shape = output_shape
        self.add_coords = AddCoords()
        in_c = in_channels + 2
        
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(50, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        
        h, w = input_spatial_shape
        bot_h, bot_w = max(1, h // 4), max(1, w // 4)
        self.bot_h, self.bot_w = bot_h, bot_w
        
        self.flatten = nn.Flatten()
        self.bot_fc = nn.Sequential(
            nn.Linear(25 * bot_h * bot_w, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 25 * bot_h * bot_w),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(nn.Conv2d(100, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.final = nn.Conv2d(25, out_channels, kernel_size=1)
        
    def _match_size(self, x, shape):
        if x.shape[-2:] != shape:
            return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        x = self.add_coords(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        b = self.pool2(e2)
        b_flat = self.flatten(b)
        b_fc = self.bot_fc(b_flat)
        b = b_fc.view(x.size(0), 25, self.bot_h, self.bot_w)
        
        d1 = self.up1(b)
        d1 = self._match_size(d1, e2.shape[-2:])
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        
        d2 = self.up2(d1)
        d2 = self._match_size(d2, e1.shape[-2:])
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        out = self.final(d2)
        return self._match_size(out, self.output_shape)

# ---------------------------------------------------------
# Config 30: Variant 3 (Residual CNN_Exp5 U-Net)
# ---------------------------------------------------------
class ResBlockCNN(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()
        
    def forward(self, x):
        res = self.skip(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + res, inplace=True)

class UNet_Config30(nn.Module):
    """
    Variant 3: Residual CNN_Exp5 U-Net
    """
    def __init__(self, in_channels, out_channels, output_shape, input_spatial_shape=(9, 10)):
        super().__init__()
        self.output_shape = output_shape
        self.add_coords = AddCoords()
        in_c = in_channels + 2
        
        self.enc1 = ResBlockCNN(in_c, 50)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = ResBlockCNN(50, 25)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.bot = ResBlockCNN(25, 25)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ResBlockCNN(50, 50)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ResBlockCNN(100, 25)
        
        self.final = nn.Conv2d(25, out_channels, kernel_size=1)
        
    def _match_size(self, x, shape):
        if x.shape[-2:] != shape:
            return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        x = self.add_coords(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))
        
        d1 = self.up1(b)
        d1 = self._match_size(d1, e2.shape[-2:])
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        
        d2 = self.up2(d1)
        d2 = self._match_size(d2, e1.shape[-2:])
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        out = self.final(d2)
        return self._match_size(out, self.output_shape)

# ---------------------------------------------------------
# Config 31: Variant 4 (Attention Gates CNN_Exp5 U-Net)
# ---------------------------------------------------------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNet_Config31(nn.Module):
    """
    Variant 4: Attention Gates CNN_Exp5 U-Net
    """
    def __init__(self, in_channels, out_channels, output_shape, input_spatial_shape=(9, 10)):
        super().__init__()
        self.output_shape = output_shape
        self.add_coords = AddCoords()
        in_c = in_channels + 2
        
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(50, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.bot = nn.Sequential(nn.Conv2d(25, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att1 = AttentionGate(F_g=25, F_l=25, F_int=12)
        self.dec1 = nn.Sequential(nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att2 = AttentionGate(F_g=50, F_l=50, F_int=25)
        self.dec2 = nn.Sequential(nn.Conv2d(100, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.final = nn.Conv2d(25, out_channels, kernel_size=1)
        
    def _match_size(self, x, shape):
        if x.shape[-2:] != shape:
            return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        x = self.add_coords(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))
        
        d1 = self.up1(b)
        d1 = self._match_size(d1, e2.shape[-2:])
        e2_att = self.att1(g=d1, x=e2)
        d1 = self.dec1(torch.cat([d1, e2_att], dim=1))
        
        d2 = self.up2(d1)
        d2 = self._match_size(d2, e1.shape[-2:])
        e1_att = self.att2(g=d2, x=e1)
        d2 = self.dec2(torch.cat([d2, e1_att], dim=1))
        
        out = self.final(d2)
        return self._match_size(out, self.output_shape)


# =========================================================
# EXPERIMENTS 32-33: ANTI-BIAS HYBRID ARCHITECTURES
# =========================================================

# ---------------------------------------------------------
# Config 32: Variant 5 (Parallel CNN_Exp5 + U-Net)
# ---------------------------------------------------------
class UNet_Config32(nn.Module):
    """
    Two branches: one exact CNN_Exp5 for global mean bias, one U-Net for spatial extremes/residuals.
    """
    def __init__(self, in_channels, out_channels, output_shape, input_spatial_shape=(9, 10)):
        super().__init__()
        self.output_shape = output_shape
        self.add_coords = AddCoords()
        in_c = in_channels + 2
        
        # Branch 1: CNN_Exp5 (Handles Bias/Mean)
        self.cnn_branch = BaselineCNNBranch(in_c, out_channels, output_shape, input_spatial_shape)
        
        # Branch 2: U-Net (Handles Extremes/Residuals)
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(50, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.bot = nn.Sequential(nn.Conv2d(25, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(nn.Conv2d(100, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.final_unet = nn.Conv2d(25, out_channels, kernel_size=1)
        
    def _match_size(self, x, shape):
        if x.shape[-2:] != shape:
            return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        x = self.add_coords(x)
        
        # Branch 1: Global map
        cnn_out = self.cnn_branch(x)
        
        # Branch 2: Local residual map
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))
        
        d1 = self.up1(b)
        d1 = self._match_size(d1, e2.shape[-2:])
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        
        d2 = self.up2(d1)
        d2 = self._match_size(d2, e1.shape[-2:])
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        unet_out = self.final_unet(d2)
        unet_out = self._match_size(unet_out, self.output_shape)
        
        # Sum both maps
        return cnn_out + unet_out

# ---------------------------------------------------------
# Config 33: Variant 6 (DenseDecoder U-Net)
# ---------------------------------------------------------
class UNet_Config33(nn.Module):
    """
    DenseDecoder: U-Net Encoder -> FC layer projecting directly to 160x170 -> Convolutions for refinement
    """
    def __init__(self, in_channels, out_channels, output_shape, input_spatial_shape=(9, 10)):
        super().__init__()
        self.output_shape = output_shape
        self.add_coords = AddCoords()
        in_c = in_channels + 2
        
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(50, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        
        h, w = input_spatial_shape
        bot_h, bot_w = max(1, h // 4), max(1, w // 4)
        out_h, out_w = output_shape
        
        self.flatten = nn.Flatten()
        
        # Dense Bottleneck -> Directly to Full Resolution Output Shape!
        self.dense_dec = nn.Sequential(
            nn.Linear(25 * bot_h * bot_w, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 25 * out_h * out_w),
            nn.ReLU(inplace=True)
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(25 + 50 + 25, 25, kernel_size=3, padding=1), # dense_map(25) + e1(50) + e2(25)
            nn.ReLU(inplace=True),
            nn.Conv2d(25, out_channels, kernel_size=1)
        )

    def _match_size(self, x, shape):
        if x.shape[-2:] != shape:
            return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        x = self.add_coords(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        b = self.pool2(e2)
        b_flat = self.flatten(b)
        
        dense_map = self.dense_dec(b_flat)
        dense_map = dense_map.view(x.size(0), 25, self.output_shape[0], self.output_shape[1])
        
        # Upsample e1 and e2 to full resolution to act as skip connections
        e1_up = self._match_size(e1, self.output_shape)
        e2_up = self._match_size(e2, self.output_shape)
        
        concat = torch.cat([dense_map, e1_up, e2_up], dim=1)
        
        out = self.refine(concat)
        return out

# ---------------------------------------------------------
# Config 34: True Residual Hybrid (Gridbox CNN + Zero-Init U-Net)
# ---------------------------------------------------------
class BaselineCNNBranch_3Channels(nn.Module):
    def __init__(self, in_channels, out_channels, output_shape, input_spatial_shape=(9, 10)):
        super().__init__()
        self.output_shape = output_shape
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        # We need this to NOT collapse to 1 channel before FC. We can just use standard flattening.
        # But wait! To keep it exactly like CNN_Exp5 but with 3 outputs:
        self.conv3 = nn.Conv2d(25, 1, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        
        fc_input_size = 1 * input_spatial_shape[0] * input_spatial_shape[1]
        
        self.fc_layers = nn.ModuleList([
            nn.Linear(fc_input_size, output_shape[0] * output_shape[1]) 
            for _ in range(out_channels)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        
        params = []
        for fc in self.fc_layers:
            p = fc(x)
            params.append(p.view(batch_size, 1, self.output_shape[0], self.output_shape[1]))
            
        return torch.cat(params, dim=1)


class UNet_Config34(nn.Module):
    """
    True Residual: CNN Base (Predicts full mean map) + UNet Base (Zero-initialized, refines extremes).
    Uses exactly 3 channels output for both, summing them cleanly.
    """
    def __init__(self, in_channels, out_channels, output_shape, input_spatial_shape=(9, 10)):
        super().__init__()
        self.output_shape = output_shape
        self.add_coords = AddCoords()
        in_c = in_channels + 2
        
        # Branch 1: CNN Anchor (Solid Baseline)
        self.cnn_branch = BaselineCNNBranch_3Channels(in_c, out_channels, output_shape, input_spatial_shape)
        
        # Branch 2: U-Net (Refinement)
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(50, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.bot = nn.Sequential(nn.Conv2d(25, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(nn.Conv2d(100, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.final_unet = nn.Conv2d(25, out_channels, kernel_size=1)
        
        # ZERO INITIALIZATION for the U-Net branch!
        nn.init.zeros_(self.final_unet.weight)
        nn.init.zeros_(self.final_unet.bias)
        
    def _match_size(self, x, shape):
        if x.shape[-2:] != shape:
            return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        x = self.add_coords(x)
        
        # Base Prediction
        cnn_out = self.cnn_branch(x)
        
        # Residual Refinement
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))
        
        d1 = self.up1(b)
        d1 = self._match_size(d1, e2.shape[-2:])
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        
        d2 = self.up2(d1)
        d2 = self._match_size(d2, e1.shape[-2:])
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        unet_out = self.final_unet(d2)
        unet_out = self._match_size(unet_out, self.output_shape)
        
        # The true addition: 3 channels + 3 channels = 3 channels
        return cnn_out + unet_out

# ---------------------------------------------------------
# Config 35: Pure 1x1 CNN Hybrid (Gridbox CNN + Zero-Init U-Net)
# ---------------------------------------------------------
class PureCNNBranch_3Channels(nn.Module):
    """
    A strictly 1x1 Convolution branch that acts as a pixel-independent non-linear GLM.
    No Dense layers, no spatial noise mixing.
    """
    def __init__(self, in_channels, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        # Interpolate first, then process pixel-by-pixel
        self.conv1 = nn.Conv2d(in_channels, 50, kernel_size=1)
        self.conv2 = nn.Conv2d(50, 25, kernel_size=1)
        self.final = nn.Conv2d(25, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=True)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.final(x)

class UNet_Config35(nn.Module):
    """
    Option A: Pure 1x1 CNN Hybrid.
    Base branch is PureCNNBranch_3Channels (No spatial mixing).
    U-Net branch is standard (Zero initialized).
    """
    def __init__(self, in_channels, out_channels, output_shape, input_spatial_shape=(9, 10)):
        super().__init__()
        self.output_shape = output_shape
        self.add_coords = AddCoords()
        in_c = in_channels + 2
        
        # Branch 1: Pure 1x1 CNN (Handles Bias/Mean locally without spreading noise)
        self.cnn_branch = PureCNNBranch_3Channels(in_c, out_channels, output_shape)
        
        # Branch 2: U-Net (Handles Extremes/Residuals)
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(50, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.bot = nn.Sequential(nn.Conv2d(25, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(50, 50, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(nn.Conv2d(100, 25, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.final_unet = nn.Conv2d(25, out_channels, kernel_size=1)
        
        # Zero-initialize the U-Net final layer to start purely from the stable CNN guess
        nn.init.zeros_(self.final_unet.weight)
        if self.final_unet.bias is not None:
            nn.init.zeros_(self.final_unet.bias)
            
    def _match_size(self, x, shape):
        if x.shape[-2:] != shape:
            return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x
        
    def forward(self, x):
        x = self.add_coords(x)
        cnn_out = self.cnn_branch(x)
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))
        
        d1 = self.dec1(torch.cat([self.up1(b), self._match_size(e2, self.up1(b).shape[-2:])], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), self._match_size(e1, self.up2(d1).shape[-2:])], dim=1))
        
        d2_up = self._match_size(d2, self.output_shape)
        unet_out = self.final_unet(d2_up)
        
        return cnn_out + unet_out

# ---------------------------------------------------------
# Config 36: CoordConv Global U-Net
# ---------------------------------------------------------
class UNet_Config36(nn.Module):
    """
    Option B: CoordConv Global U-Net.
    Standard U-Net architecture. Uses Global Normalization for perfect LMDZ stability.
    Relies on AddCoords (Latitude/Longitude) to deduce topography.
    """
    def __init__(self, in_channels, out_channels, output_shape, input_spatial_shape=(9, 10)):
        super().__init__()
        self.output_shape = output_shape
        self.add_coords = AddCoords()
        in_c = in_channels + 2
        
        self.enc1 = nn.Sequential(nn.Conv2d(in_c, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.bot = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def _match_size(self, x, shape):
        if x.shape[-2:] != shape:
            return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x
        
    def forward(self, x):
        x = self.add_coords(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        b = self.bot(self.pool2(e2))
        
        d1 = self.dec1(torch.cat([self.up1(b), self._match_size(e2, self.up1(b).shape[-2:])], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), self._match_size(e1, self.up2(d1).shape[-2:])], dim=1))
        
        d2_up = self._match_size(d2, self.output_shape)
        return self.final(d2_up)
