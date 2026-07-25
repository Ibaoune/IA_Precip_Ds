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
