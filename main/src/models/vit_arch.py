import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ==========================================
# Device Configuration
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# Loss Function
# ==========================================
class BernoulliGammaLoss(nn.Module):
    """
    Custom loss function tailored for precipitation downscaling.
    Models rainfall as a mixed discrete-continuous process:
    - Bernoulli distribution for rain occurrence.
    - Gamma distribution for rainfall intensity.
    """
    def __init__(self, eps: float = 1e-6):
        super(BernoulliGammaLoss, self).__init__()
        self.eps = eps

    def forward(self, true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        # Extract parameters and apply constraints to prevent numerical instability
        occurrence = torch.sigmoid(pred[:, 0, :, :]).clamp(self.eps, 1 - self.eps)
        shape_parameter = torch.exp(pred[:, 1, :, :].clamp(-5, 5)).clamp(self.eps, 1e3)
        scale_parameter = torch.exp(pred[:, 2, :, :].clamp(-5, 5)).clamp(self.eps, 1e3)
        
        # GPU-safe boolean mask for rainfall occurrence
        bool_rain = (true > 0).float()

        # Calculate negative log-likelihood parts
        no_rain_loss = (1 - bool_rain) * torch.log(1 - occurrence + self.eps)
        
        rain_loss = bool_rain * (
            torch.log(occurrence + self.eps) + 
            (shape_parameter - 1) * torch.log(true + self.eps) - 
            shape_parameter * torch.log(scale_parameter + self.eps) - 
            torch.lgamma(shape_parameter + self.eps) - 
            (true / (scale_parameter + self.eps))
        )

        # Final loss is the negative mean of the combined probabilities
        loss = -torch.mean(no_rain_loss + rain_loss)
        
        return loss

# ==========================================
# Transformer Components
# ==========================================
class PatchEmbedding(nn.Module):
    """
    Splits the 2D input grid into non-overlapping patches and projects 
    them into a linear embedding space.
    """
    def __init__(self, in_channels: int, patch_size: int, emb_size: int):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels, emb_size, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: torch.Tensor):
        patches = self.projection(x)
        h_patches, w_patches = patches.shape[2], patches.shape[3]
        
        # Flatten the spatial dimensions: (batch, channels, height, width) -> (batch, num_patches, emb_size)
        patches = rearrange(patches, 'b e h w -> b (h w) e')
        return patches, h_patches, w_patches


class TransformerBlock(nn.Module):
    """
    A single Vision Transformer block featuring Multi-Head Self Attention 
    and a Feed-Forward Network with GELU activation.
    """
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.GELU(),
            nn.Linear(emb_size * 4, emb_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention mechanism
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward network
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Stacks multiple Transformer blocks to form the encoder.
    """
    def __init__(self, emb_size: int, num_layers: int, num_heads: int, dropout: float):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(emb_size, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

# ==========================================
# Decoder Component
# ==========================================
class UpsamplingDecoder(nn.Module):
    """
    Reconstructs the spatial grid from the encoded sequence of patches 
    using transposed convolution, and pads to the exact target dimensions if necessary.
    """
    def __init__(self, emb_size: int, patch_size: int, output_channels: int):
        super(UpsamplingDecoder, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.ConvTranspose2d(
            emb_size, output_channels,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor, h_patches: int, w_patches: int, target_h: int = None, target_w: int = None) -> torch.Tensor:
        # Reshape back to 2D format: (batch, num_patches, emb_size) -> (batch, emb_size, height, width)
        x = rearrange(x, 'b (h w) e -> b e h w', h=h_patches, w=w_patches)
        x = self.projection(x)

        # Pad dynamically if reconstructed image is slightly smaller than target targets
        if target_h is not None and target_w is not None:
            pad_h = target_h - x.shape[2]
            pad_w = target_w - x.shape[3]
            if pad_h > 0 or pad_w > 0:
               x = F.pad(x, (0, pad_w, 0, pad_h))
               
        return x

# ==========================================
# Full Model Architecture
# ==========================================
class DownscalingViT(nn.Module):
    """
    Complete Vision Transformer for 2D data downscaling.
    Transforms low-resolution inputs into high-resolution targets.
    """
    def __init__(self, in_channels: int, emb_size: int, patch_size: int, num_layers: int, 
                 num_heads: int, dropout: float, output_channels: int, n_lat_out: int, n_lon_out: int):
        super(DownscalingViT, self).__init__()

        # Target dimensions
        self.n_lat_out = n_lat_out
        self.n_lon_out = n_lon_out

        # Core modules
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size)
        self.transformer = TransformerEncoder(emb_size, num_layers, num_heads, dropout)
        self.decoder = UpsamplingDecoder(emb_size, patch_size, output_channels)

        # Calculate number of patches strictly during initialization (GPU-Safe Pattern)
        h_patches = n_lat_out // patch_size
        w_patches = n_lon_out // patch_size
        num_patches = h_patches * w_patches

        # GPU ADAPTATION: Initialize Positional Embedding as a registered nn.Parameter.
        # This ensures it is tracked by the model, so when model.to(cfg.device) is called,
        # the positional embedding is correctly moved to the GPU along with the weights.
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, emb_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape expected: (batch_size, in_channels, height, width)
        """
        # Step 1: Interpolate input to target spatial dimensions
        x = F.interpolate(x, size=(self.n_lat_out, self.n_lon_out), mode='nearest')

        # Step 2: Extract and flatten patches
        patches, h_patches, w_patches = self.patch_embedding(x)

        # Step 3: Add learned positional embeddings
        patches = patches + self.positional_embedding

        # Step 4: Pass through the Transformer Encoder
        encoded_patches = self.transformer(patches)

        # Step 5: Decode back to image space
        out = self.decoder(
            encoded_patches, 
            h_patches, 
            w_patches,
            target_h=self.n_lat_out, 
            target_w=self.n_lon_out
        )
        
        return out