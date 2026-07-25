import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from src.models.vit_arch import PatchEmbedding, TransformerEncoder, UpsamplingDecoder, get_2d_sincos_pos_embed

# ---------------------------------------------------------
# Exp 1: PatchEmbedding sur la Basse Résolution (Early Patching)
# ---------------------------------------------------------
class ViT_Exp1(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size, num_layers, num_heads, dropout, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.out_channels = out_channels
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, stride=patch_size)
        self.transformer = TransformerEncoder(emb_size, num_layers, num_heads, dropout)
        # Le décodeur devra franchir un plus grand gap (low res -> high res)
        self.decoder = UpsamplingDecoder(emb_size, patch_size, out_channels, stride=patch_size)

    def forward(self, x):
        # NO INTERPOLATION AT START!
        patches, h_patches, w_patches = self.patch_embedding(x)
        encoded_patches = self.transformer(patches)
        x_out = self.decoder(encoded_patches, h_patches, w_patches, target_h=self.output_shape[0], target_w=self.output_shape[1])
        
        # Final interpolation just to be safe if decoder doesn't exactly match
        if x_out.shape[-2:] != self.output_shape:
            x_out = F.interpolate(x_out, size=self.output_shape, mode='bilinear', align_corners=True)
            
        if self.out_channels == 3:
            x1 = torch.sigmoid(x_out[:, 0:1, :, :])
            x2 = F.softplus(x_out[:, 1:2, :, :])
            x3 = F.softplus(x_out[:, 2:3, :, :])
            x_out = torch.cat([x1, x2, x3], dim=1)
        return x_out

# ---------------------------------------------------------
# Exp 2: Décodeur "Dense" (ViT-to-CNN)
# ---------------------------------------------------------
class ViT_Exp2(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size, num_layers, num_heads, dropout, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.out_channels = out_channels
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, stride=patch_size)
        self.transformer = TransformerEncoder(emb_size, num_layers, num_heads, dropout)
        
        # Compute exact max sequence length approx to define linear layer size
        # To make it input shape agnostic, we use adaptive pooling over the 1D sequence or 2D map
        self.fc_layers = nn.ModuleList([
            nn.Linear(emb_size, np.prod(output_shape)) 
            for _ in range(out_channels)
        ])
        
    def forward(self, x):
        x = F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)
        patches, h_patches, w_patches = self.patch_embedding(x)
        encoded_patches = self.transformer(patches)
        
        # Global Average Pooling over the sequence length
        # encoded_patches: (batch, seq_len, emb_size) -> (batch, emb_size)
        pooled = encoded_patches.mean(dim=1)
        
        params = []
        for fc in self.fc_layers:
            p = fc(pooled)
            params.append(p.view(x.size(0), 1, self.output_shape[0], self.output_shape[1]))
            
        x_out = torch.cat(params, dim=1)
        
        if self.out_channels == 3:
            x1 = torch.sigmoid(x_out[:, 0:1, :, :])
            x2 = F.softplus(x_out[:, 1:2, :, :])
            x3 = F.softplus(x_out[:, 2:3, :, :])
            x_out = torch.cat([x1, x2, x3], dim=1)
        return x_out

# ---------------------------------------------------------
# Exp 3: Patchs avec Chevauchement (Overlapping Patches)
# ---------------------------------------------------------
class ViT_Exp3(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size, num_layers, num_heads, dropout, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.out_channels = out_channels
        
        # Stride = patch_size // 2 for overlap
        stride = max(1, patch_size // 2)
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, stride=stride)
        self.transformer = TransformerEncoder(emb_size, num_layers, num_heads, dropout)
        self.decoder = UpsamplingDecoder(emb_size, patch_size, out_channels, stride=stride)

    def forward(self, x):
        x = F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)
        patches, h_patches, w_patches = self.patch_embedding(x)
        encoded_patches = self.transformer(patches)
        x_out = self.decoder(encoded_patches, h_patches, w_patches, target_h=self.output_shape[0], target_w=self.output_shape[1])
        
        if self.out_channels == 3:
            x1 = torch.sigmoid(x_out[:, 0:1, :, :])
            x2 = F.softplus(x_out[:, 1:2, :, :])
            x3 = F.softplus(x_out[:, 2:3, :, :])
            x_out = torch.cat([x1, x2, x3], dim=1)
        return x_out

# ---------------------------------------------------------
# Exp 4: Décodeur CNN Progressif (ViT-UNet Hybrid)
# ---------------------------------------------------------
class ViT_Exp4(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size, num_layers, num_heads, dropout, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.out_channels = out_channels
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, stride=patch_size)
        self.transformer = TransformerEncoder(emb_size, num_layers, num_heads, dropout)
        
        self.conv1 = nn.Conv2d(emb_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)
        patches, h_patches, w_patches = self.patch_embedding(x)
        encoded_patches = self.transformer(patches)
        
        # Reshape to 2D
        encoded_2d = rearrange(encoded_patches, 'b (h w) e -> b e h w', h=h_patches, w=w_patches)
        
        # Progressive Upsampling
        up1 = F.interpolate(encoded_2d, scale_factor=2, mode='bilinear', align_corners=True)
        up1 = F.relu(self.conv1(up1))
        
        x_out = F.interpolate(up1, size=self.output_shape, mode='bilinear', align_corners=True)
        x_out = self.conv2(x_out)
        
        if self.out_channels == 3:
            x1 = torch.sigmoid(x_out[:, 0:1, :, :])
            x2 = F.softplus(x_out[:, 1:2, :, :])
            x3 = F.softplus(x_out[:, 2:3, :, :])
            x_out = torch.cat([x1, x2, x3], dim=1)
        return x_out

# ---------------------------------------------------------
# Exp 5: Skip Connection Globale (Apprentissage Résiduel)
# ---------------------------------------------------------
class ViT_Exp5(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size, num_layers, num_heads, dropout, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.out_channels = out_channels
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, stride=patch_size)
        self.transformer = TransformerEncoder(emb_size, num_layers, num_heads, dropout)
        self.decoder = UpsamplingDecoder(emb_size, patch_size, out_channels, stride=patch_size)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x_resized = F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)
        patches, h_patches, w_patches = self.patch_embedding(x_resized)
        encoded_patches = self.transformer(patches)
        x_out = self.decoder(encoded_patches, h_patches, w_patches, target_h=self.output_shape[0], target_w=self.output_shape[1])
        
        # Residual mapping
        res = self.skip_conv(x_resized)
        x_out = x_out + res
        
        if self.out_channels == 3:
            x1 = torch.sigmoid(x_out[:, 0:1, :, :])
            x2 = F.softplus(x_out[:, 1:2, :, :])
            x3 = F.softplus(x_out[:, 2:3, :, :])
            x_out = torch.cat([x1, x2, x3], dim=1)
        return x_out

# ---------------------------------------------------------
# Exp 6: ViT avec CoordConv
# ---------------------------------------------------------
from src.models.unet_arch import AddCoords
class ViT_Exp6(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size, num_layers, num_heads, dropout, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.out_channels = out_channels
        self.add_coords = AddCoords()
        self.patch_embedding = PatchEmbedding(in_channels + 2, patch_size, emb_size, stride=patch_size)
        self.transformer = TransformerEncoder(emb_size, num_layers, num_heads, dropout)
        self.decoder = UpsamplingDecoder(emb_size, patch_size, out_channels, stride=patch_size)

    def forward(self, x):
        x = F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)
        x = self.add_coords(x)
        patches, h_patches, w_patches = self.patch_embedding(x)
        encoded_patches = self.transformer(patches)
        x_out = self.decoder(encoded_patches, h_patches, w_patches, target_h=self.output_shape[0], target_w=self.output_shape[1])
        
        if self.out_channels == 3:
            x1 = torch.sigmoid(x_out[:, 0:1, :, :])
            x2 = F.softplus(x_out[:, 1:2, :, :])
            x3 = F.softplus(x_out[:, 2:3, :, :])
            x_out = torch.cat([x1, x2, x3], dim=1)
        return x_out

# ---------------------------------------------------------
# Exp 7: Attention Masking Spatial (TBD/Approximation)
# ---------------------------------------------------------
# Simplified version of restricted attention via shorter sequence length (larger patch sizes)
class ViT_Exp7(ViT_Exp1):
    def __init__(self, in_channels, emb_size, patch_size, num_layers, num_heads, dropout, out_channels, output_shape):
        # Use a much larger patch size to enforce local grouping
        super().__init__(in_channels, emb_size, patch_size * 2, num_layers, num_heads, dropout, out_channels, output_shape)

# ---------------------------------------------------------
# Exp 8: ViT Hybride (CNN Backbone)
# ---------------------------------------------------------
class ViT_Exp8(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size, num_layers, num_heads, dropout, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.out_channels = out_channels
        
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.patch_embedding = PatchEmbedding(16, patch_size, emb_size, stride=patch_size)
        self.transformer = TransformerEncoder(emb_size, num_layers, num_heads, dropout)
        self.decoder = UpsamplingDecoder(emb_size, patch_size, out_channels, stride=patch_size)

    def forward(self, x):
        x = F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)
        x = self.cnn_backbone(x)
        patches, h_patches, w_patches = self.patch_embedding(x)
        encoded_patches = self.transformer(patches)
        x_out = self.decoder(encoded_patches, h_patches, w_patches, target_h=self.output_shape[0], target_w=self.output_shape[1])
        
        if self.out_channels == 3:
            x1 = torch.sigmoid(x_out[:, 0:1, :, :])
            x2 = F.softplus(x_out[:, 1:2, :, :])
            x3 = F.softplus(x_out[:, 2:3, :, :])
            x_out = torch.cat([x1, x2, x3], dim=1)
        return x_out
