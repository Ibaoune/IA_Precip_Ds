import os

new_code = """

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
    \"\"\"
    Two branches: one exact CNN for global mean bias, one standard U-Net for spatial extremes.
    \"\"\"
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
    \"\"\"
    A U-Net where the bottleneck passes through a Linear layer, 
    allowing for a global shift to be learned at the lowest resolution.
    \"\"\"
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
    \"\"\"
    Standard U-Net but adds a parallel branch that applies GAP + MLP to output a global spatial bias map.
    \"\"\"
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
    \"\"\"
    Uses the exact CNN 3-layer architecture as the encoder (no pooling), 
    and a decoder to project it directly to the output resolution.
    \"\"\"
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
    \"\"\"
    Combines Dilated UNet (Config 18) and CNN with a learnable gate.
    \"\"\"
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
"""

with open("main/src/models/unet_experiments.py", "a") as f:
    f.write(new_code)
