import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Standard Convolutional Block (k=3, padding=1) with BN/GN and ReLU"""
    def __init__(self, in_channels, out_channels, group_norm_enable=False, num_groups=32, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels) if group_norm_enable else nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels) if group_norm_enable else nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.norm2(self.conv2(x)))
        return x

class UpSampleBlock(nn.Module):
    """Bilinear upsampling followed by a 3x3 convolution to reduce channels (avoids checkerboard)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.up(x)

class AttentionBlock(nn.Module):
    """Attention Gate for filtering skip connections"""
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(f_int))
        self.W_x = nn.Sequential(nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Make sure dims match
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
            
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AddCoords(nn.Module):
    """Inject normalized (x,y) spatial coordinates to break translation invariance"""
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()
        
        xx_channel = torch.arange(x_dim, dtype=torch.float32, device=input_tensor.device)
        yy_channel = torch.arange(y_dim, dtype=torch.float32, device=input_tensor.device)

        xx_channel = (xx_channel / (x_dim - 1)) * 2 - 1
        yy_channel = (yy_channel / (y_dim - 1)) * 2 - 1

        xx_channel = xx_channel.view(1, 1, x_dim, 1).expand(batch_size, 1, x_dim, y_dim)
        yy_channel = yy_channel.view(1, 1, 1, y_dim).expand(batch_size, 1, x_dim, y_dim)

        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)
        return ret


# =====================================================================================
# Legacy Flawed UNet (Kept for compatibility, though discouraged)
# =====================================================================================
class UNet(nn.Module):
    def __init__(self, in_channels=15, out_channels=1, group_norm_enable=False, num_groups=32):
        super().__init__()
        self.gn_enable = group_norm_enable
        self.num_groups = num_groups

        self.enc_conv1 = self.conv_block(in_channels, 64, k=2)
        self.enc_conv2 = self.conv_block(64, 128, k=2)
        self.enc_conv3 = self.conv_block(128, 256, k=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck_conv = self.conv_block(256, 512, k=2)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = self.conv_block(512, 256, k=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = self.conv_block(256, 128, k=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = self.conv_block(128, 64, k=2)

        self.final_output = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)

    def conv_block(self, in_channels, out_channels, k=2):
        norm1 = nn.GroupNorm(self.num_groups, out_channels) if self.gn_enable else nn.BatchNorm2d(out_channels)
        norm2 = nn.GroupNorm(self.num_groups, out_channels) if self.gn_enable else nn.BatchNorm2d(out_channels)
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=1, padding='same'), norm1, nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=k, stride=1, padding='same'), norm2, nn.ReLU(inplace=True),
        )

    @staticmethod
    def _match_size(upsampled, skip):
        diff_h = skip.size(2) - upsampled.size(2)
        diff_w = skip.size(3) - upsampled.size(3)
        return F.pad(upsampled, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

    def forward(self, x):
        c1 = self.enc_conv1(x); p1 = self.pool(c1)
        c2 = self.enc_conv2(p1); p2 = self.pool(c2)
        c3 = self.enc_conv3(p2); p3 = self.pool(c3)
        bn = self.bottleneck_conv(p3)

        u3 = self._match_size(self.upconv3(bn), c3)
        d3 = self.dec_conv3(torch.cat([u3, c3], dim=1))
        u2 = self._match_size(self.upconv2(d3), c2)
        d2 = self.dec_conv2(torch.cat([u2, c2], dim=1))
        u1 = self._match_size(self.upconv1(d2), c1)
        d1 = self.dec_conv1(torch.cat([u1, c1], dim=1))

        return self.final_output(d1)


# =====================================================================================
# UNet V2: Fixed Kernel Size (3x3), Bilinear Upsample, and Dropout Regularization
# =====================================================================================
class UNet_V2(nn.Module):
    def __init__(self, in_channels=15, out_channels=1, group_norm_enable=False, num_groups=32, dropout_p=0.1):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64, group_norm_enable, num_groups, dropout_p)
        self.enc2 = ConvBlock(64, 128, group_norm_enable, num_groups, dropout_p)
        self.enc3 = ConvBlock(128, 256, group_norm_enable, num_groups, dropout_p)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(256, 512, group_norm_enable, num_groups, dropout_p)

        self.up3 = UpSampleBlock(512, 256)
        self.dec3 = ConvBlock(512, 256, group_norm_enable, num_groups, dropout_p)
        
        self.up2 = UpSampleBlock(256, 128)
        self.dec2 = ConvBlock(256, 128, group_norm_enable, num_groups, dropout_p)
        
        self.up1 = UpSampleBlock(128, 64)
        self.dec1 = ConvBlock(128, 64, group_norm_enable, num_groups, dropout_p)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def _match_size(self, upsampled, skip):
        if upsampled.shape != skip.shape:
            upsampled = F.interpolate(upsampled, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return upsampled

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        bn = self.bottleneck(self.pool(e3))
        
        d3 = self.dec3(torch.cat([self._match_size(self.up3(bn), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match_size(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match_size(self.up1(d2), e1), e1], dim=1))
        
        return self.final(d1)


# =====================================================================================
# UNet CoordConv: Breaks Translation Invariance by injecting absolute coordinates
# =====================================================================================
class UNet_CoordConv(nn.Module):
    def __init__(self, in_channels=15, out_channels=1, group_norm_enable=False, num_groups=32, dropout_p=0.1):
        super().__init__()
        self.add_coords = AddCoords()
        # +2 for the (x, y) coordinates injected
        self.unet = UNet_V2(in_channels + 2, out_channels, group_norm_enable, num_groups, dropout_p)

    def forward(self, x):
        x_coord = self.add_coords(x)
        return self.unet(x_coord)


# =====================================================================================
# Attention UNet: Filters irrelevant background from skip connections
# =====================================================================================
class Attention_UNet(nn.Module):
    def __init__(self, in_channels=15, out_channels=1, group_norm_enable=False, num_groups=32, dropout_p=0.1):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64, group_norm_enable, num_groups, dropout_p)
        self.enc2 = ConvBlock(64, 128, group_norm_enable, num_groups, dropout_p)
        self.enc3 = ConvBlock(128, 256, group_norm_enable, num_groups, dropout_p)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(256, 512, group_norm_enable, num_groups, dropout_p)

        self.up3 = UpSampleBlock(512, 256)
        self.att3 = AttentionBlock(f_g=256, f_l=256, f_int=128)
        self.dec3 = ConvBlock(512, 256, group_norm_enable, num_groups, dropout_p)
        
        self.up2 = UpSampleBlock(256, 128)
        self.att2 = AttentionBlock(f_g=128, f_l=128, f_int=64)
        self.dec2 = ConvBlock(256, 128, group_norm_enable, num_groups, dropout_p)
        
        self.up1 = UpSampleBlock(128, 64)
        self.att1 = AttentionBlock(f_g=64, f_l=64, f_int=32)
        self.dec1 = ConvBlock(128, 64, group_norm_enable, num_groups, dropout_p)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def _match_size(self, upsampled, skip):
        if upsampled.shape != skip.shape:
            upsampled = F.interpolate(upsampled, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return upsampled

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        bn = self.bottleneck(self.pool(e3))
        
        # Up 3
        g3 = self.up3(bn)
        x3 = self._match_size(g3, e3)
        att3_out = self.att3(g=x3, x=e3)
        d3 = self.dec3(torch.cat([x3, att3_out], dim=1))
        
        # Up 2
        g2 = self.up2(d3)
        x2 = self._match_size(g2, e2)
        att2_out = self.att2(g=x2, x=e2)
        d2 = self.dec2(torch.cat([x2, att2_out], dim=1))
        
        # Up 1
        g1 = self.up1(d2)
        x1 = self._match_size(g1, e1)
        att1_out = self.att1(g=x1, x=e1)
        d1 = self.dec1(torch.cat([x1, att1_out], dim=1))
        
        return self.final(d1)

# =====================================================================================
# Doury UNet: Deeper architecture with ELU activation designed for RCM emulation
# =====================================================================================
class ELUConvBlock(nn.Module):
    """Convolutional Block using ELU (Exponential Linear Unit) for climate variables"""
    def __init__(self, in_channels, out_channels, group_norm_enable=False, num_groups=32, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels) if group_norm_enable else nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels) if group_norm_enable else nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        x = self.elu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.elu(self.norm2(self.conv2(x)))
        return x

class Doury_UNet(nn.Module):
    """
    Architecture inspired by Doury et al. (2024) for fine spatio-temporal precipitation.
    Adapted for 3 downsampling layers to support coarser grid resolutions.
    Deep features, Bilinear upsampling, and ELU activations to avoid 'dying' gradients on 0 values.
    """
    def __init__(self, in_channels=15, out_channels=1, group_norm_enable=False, num_groups=32, dropout_p=0.1):
        super().__init__()
        self.add_coords = AddCoords()
        in_c = in_channels + 2
        
        self.enc1 = ELUConvBlock(in_c, 64, group_norm_enable, num_groups, dropout_p)
        self.enc2 = ELUConvBlock(64, 128, group_norm_enable, num_groups, dropout_p)
        self.enc3 = ELUConvBlock(128, 256, group_norm_enable, num_groups, dropout_p)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = ELUConvBlock(256, 512, group_norm_enable, num_groups, dropout_p)

        self.up3 = UpSampleBlock(512, 256)
        self.dec3 = ELUConvBlock(512, 256, group_norm_enable, num_groups, dropout_p)
        
        self.up2 = UpSampleBlock(256, 128)
        self.dec2 = ELUConvBlock(256, 128, group_norm_enable, num_groups, dropout_p)
        
        self.up1 = UpSampleBlock(128, 64)
        self.dec1 = ELUConvBlock(128, 64, group_norm_enable, num_groups, dropout_p)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def _match_size(self, upsampled, skip):
        if upsampled.shape != skip.shape:
            upsampled = F.interpolate(upsampled, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return upsampled

    def forward(self, x):
        x = self.add_coords(x)
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        bn = self.bottleneck(self.pool(e3))
        
        d3 = self.dec3(torch.cat([self._match_size(self.up3(bn), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match_size(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match_size(self.up1(d2), e1), e1], dim=1))
        
        return self.final(d1)