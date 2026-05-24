import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import warnings

# ==========================================
# Device Configuration
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# Loss Function (From lit_version/models/losses.py)
# ==========================================
class BernoulliGammaLoss(nn.Module):
    def __init__(self, reduction="mean", eps=1e-3):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, y, mask=None):
        """
        pi:    (B, ...) Bernoulli probability, in (0,1)
        alpha: (B, ...) Gamma shape > 0
        beta:  (B, ...) Gamma scale > 0
        y:     (B, ...) target values, either 0 or strictly > 0
        """
        # Compatibility check if called with (true, pred) order
        if y.dim() > 1 and y.size(1) == 3 and pred.size(1) != 3:
            pred, y = y, pred

        if y.dim() == 4 and y.size(1) == 1:
            y = y.squeeze(1)

        pi = pred[:, 0]
        alpha = pred[:, 1]
        beta = pred[:, 2]

        if torch.any(y < - self.eps):
            warnings.warn(
                "Some target values are negative, check if any normalisation is applied"
                f"min={y.min()}, max={y.max()}", UserWarning)
        # Runtime checks with warnings
        if torch.any((pi < 0) | (pi > 1)):
            warnings.warn(
                "Some pi values are outside (0,1). They will be clamped."
                f"min={pi.detach().min().item()}, max={pi.detach().max().item()}", UserWarning)
            pi = torch.clamp(pi, 1e-6, 1 - self.eps)
        if torch.any(alpha < 0):
            warnings.warn(
                "Some alpha values are <= 0. They will be clamped."
                f"min={alpha.detach().min().item()}, max={alpha.detach().max().item()}", UserWarning)
            alpha = torch.clamp(alpha, 0, None)
        if torch.any(beta < 1e-6):
            warnings.warn(
                f"Some beta values are <= {1e-6}. They will be clamped."
                f"min={beta.detach().min().item()}, max={beta.detach().max().item()}", UserWarning)
            beta = torch.clamp(beta, self.eps, None)

        # Case y == 0
        loss_zero = -torch.log1p(-pi)
        loss_pos = (
            - torch.log(pi)
            + torch.lgamma(alpha)
            + alpha * torch.log(beta)
            - (alpha - 1) * torch.log(torch.clamp(y, min=self.eps))
            + y / beta
        )

        occurence_mask = (y > self.eps).float()
        loss_elementwise = (1 - occurence_mask) * loss_zero + occurence_mask * loss_pos

        if mask is not None:
            return (loss_elementwise * mask).sum() / (mask.sum() * pred.size(0))

        if self.reduction == "mean":
            return loss_elementwise.mean()
        elif self.reduction == "sum":
            return loss_elementwise.sum()
        else:
            return loss_elementwise


# ==========================================
# Transformer Components (From lit_version/models/vit.py)
# ==========================================
def get_2d_sincos_pos_embed(embed_dim, h, w, device):
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )

    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"

    omega = torch.arange(embed_dim // 4, device=device) / (embed_dim // 4)
    omega = 1. / (10000 ** omega)

    out_y = torch.einsum('hw,d->hwd', grid_y, omega)
    out_x = torch.einsum('hw,d->hwd', grid_x, omega)

    pos_emb = torch.cat(
        [torch.sin(out_y), torch.cos(out_y),
         torch.sin(out_x), torch.cos(out_x)],
        dim=-1
    )

    return pos_emb.reshape(h * w, embed_dim)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, stride=None):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        stride = stride or patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        patches = self.projection(x)
        h_patches, w_patches = patches.shape[2], patches.shape[3]
        patches = rearrange(patches, 'b e h w -> b (h w) e')
        # positional embeddings
        pos_emb = get_2d_sincos_pos_embed(patches.shape[-1], h_patches, w_patches, x.device)
        patches = patches + pos_emb.unsqueeze(0)
        return patches, h_patches, w_patches


class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
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

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, emb_size, num_layers, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(emb_size, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class UpsamplingDecoder(nn.Module):
    def __init__(self, emb_size, patch_size, out_channels, stride=None):
        """
        Upsamples the encoded representation back into an image.

        emb_size (int): Dimension of each patch
        patch_size (int): Size of each square patch
        out_channels (int): Number of output channels
        """
        super(UpsamplingDecoder, self).__init__()
        self.patch_size = patch_size
        stride = stride or patch_size
        self.projection = nn.ConvTranspose2d(
            emb_size, out_channels,
            kernel_size=patch_size, stride=stride
        )

    def forward(self, x, h_patches, w_patches, target_h=None, target_w=None):
        x = rearrange(x, 'b (h w) e -> b e h w', h=h_patches, w=w_patches)
        x = self.projection(x)

        if target_h is not None and target_w is not None:
            pad_h = target_h - x.shape[-2]
            pad_w = target_w - x.shape[-1]
            if pad_h > 0 or pad_w > 0:
               x = F.pad(x, (0, pad_w, 0, pad_h))
        return x


# Modèle complet combinant embedding, encodeur Transformer et décodeur
class DownscalingViT(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size, num_layers, num_heads,
                 dropout=0.1, out_channels=None, output_shape=None, stride=None, forcings_dim=0,
                 output_channels=None, n_lat_out=None, n_lon_out=None):
        """
        Vision Transformer pour des tâches de descente d’échelle 2D.
        """
        super(DownscalingViT, self).__init__()

        # Compatibility between main and lit_version parameter naming
        out_ch = out_channels if out_channels is not None else output_channels
        if out_ch is None:
            out_ch = 1

        if output_shape is not None:
            self.output_shape = output_shape
        else:
            self.output_shape = (n_lat_out, n_lon_out)

        # Module d'encodage des patchs
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, stride=stride)
        
        self.forcings_dim = forcings_dim
        if forcings_dim > 0:
            self.fc_forcings = nn.Linear(forcings_dim, emb_size)
        # Encodeur Transformer
        self.transformer = TransformerEncoder(emb_size, num_layers, num_heads, dropout)

        # Décodeur final
        self.decoder = UpsamplingDecoder(emb_size, patch_size, out_ch, stride=stride)
        self.out_ch = out_ch

    def forward(self, x, forcings=None):
        # x : (batch_size, in_channels, height, width)
        # Redimensionne l’entrée à la taille cible par interpolation bilinear
        x = nn.functional.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)

        # Découpe l’image en patchs encodés
        patches, h_patches, w_patches = self.patch_embedding(x)

        # Ajout des forcings saisonniers si fournis
        if forcings is not None and self.forcings_dim > 0:
            forcings_emb = self.fc_forcings(forcings)  # (batch_size, emb_size)
            patches = patches + forcings_emb.unsqueeze(1)  # Broadcasting

        # Passage dans l’encodeur Transformer
        encoded_patches = self.transformer(patches)

        # Reconstruction finale de l’image
        x = self.decoder(encoded_patches, h_patches, w_patches,
                 target_h=self.output_shape[0], target_w=self.output_shape[1])
        # if out_ch == 3, apply activation functions
        if self.out_ch == 3:
            # First channel: ocurrence (sigmoid)
            x1 = torch.sigmoid(x[:, 0:1, :, :])
            # Second channel: shape_parameter (softplus)
            x2 = F.softplus(x[:, 1:2, :, :])
            # Third channel: scale_parameter (softplus)
            x3 = F.softplus(x[:, 2:3, :, :])
            x = torch.cat([x1, x2, x3], dim=1)
        return x


if __name__ == "__main__":
    # Test the model with random input
    model = DownscalingViT(
        in_channels=4,
        emb_size=128,
        patch_size=(4, 6),
        num_layers=4,
        num_heads=4,
        dropout=0.0,
        out_channels=3,
        output_shape=(70, 100),
        stride=(2, 3)
    )
    x = torch.randn(2, 4, 35, 50)  # batch_size=2, in_channels=4, height=35, width=50
    out = model(x)
    print(out.shape)  # Expected output shape: (2, 3, 70, 100)