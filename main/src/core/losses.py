import torch
import torch.nn as nn

class BernoulliGammaLoss(nn.Module):
    def __init__(self):
        super(BernoulliGammaLoss, self).__init__()

    def forward(self, pred, true):
        """
        pred: (B, 3, H, W) - [occurrence, shape, scale]
        true: (B, 1, H, W) - target precipitation
        """
        eps = 1e-5
        
        if true.dim() == 4 and true.size(1) == 1:
            true = true.squeeze(1)
            
        occurrence = torch.sigmoid(pred[:, 0, :, :]).clamp(eps, 1 - eps)
        shape_parameter = torch.exp(pred[:, 1, :, :].clamp(-10, 7)).clamp(eps, 1e3)
        scale_parameter = torch.exp(pred[:, 2, :, :].clamp(-10, 7)).clamp(eps, 1e3)
        
        bool_rain = (true > 0).float()
        epsilon = 1e-6

        loss = (-torch.mean((1 - bool_rain) * torch.log(1 - occurrence + epsilon) + 
                             bool_rain * (torch.log(occurrence + epsilon) + 
                                          (shape_parameter - 1) * torch.log(true + epsilon) -
                                          shape_parameter * torch.log(scale_parameter + epsilon) -
                                          torch.lgamma(shape_parameter + epsilon) -
                                          true / (scale_parameter + epsilon))))

        return loss

import math

class GaussianLoss(nn.Module):
    def __init__(self):
        super(GaussianLoss, self).__init__()

    def forward(self, pred, true):
        """
        pred: (B, 2, H, W) - [mean, log_var]
        true: (B, 1, H, W) - target temperature
        """
        if true.dim() == 4 and true.size(1) == 1:
            true = true.squeeze(1)
            
        mean = pred[:, 0, :, :]
        log_var = pred[:, 1, :, :] # This is ln(sigma^2)
        
        loss = 0.5 * torch.mean(math.log(2 * math.pi) + log_var + torch.exp(-log_var) * (true - mean)**2)
        return loss

import torch.nn.functional as F

class AsymmetricMSELoss(nn.Module):
    """
    Penalizes underestimating heavy rainfall more than overestimating light rainfall.
    Inspired by Doury et al. (2024).
    """
    def __init__(self, alpha=3.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, true):
        if true.dim() == 4 and true.size(1) == 1:
            true = true.squeeze(1)
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
            
        diff = true - pred
        # If true > pred (underestimation), multiply squared error by alpha
        loss = torch.where(diff > 0, self.alpha * (diff ** 2), diff ** 2)
        return torch.mean(loss)

class IntensityWeightedMSELoss(nn.Module):
    """
    Scales the MSE loss by the true intensity: Loss = (1 + w * y_true) * (y_true - y_pred)^2
    """
    def __init__(self, weight_factor=1.0):
        super().__init__()
        self.w = weight_factor

    def forward(self, pred, true):
        if true.dim() == 4 and true.size(1) == 1:
            true = true.squeeze(1)
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
            
        mse = (true - pred) ** 2
        weights = 1.0 + self.w * true
        return torch.mean(weights * mse)

class HurdleLoss(nn.Module):
    """
    Two-part loss for zero-inflated continuous data.
    Channel 0: Probability of rain (BCE)
    Channel 1: Intensity of rain given that it rains (L1 or Asymmetric MSE)
    """
    def __init__(self, alpha=3.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, true):
        if true.dim() == 4 and true.size(1) == 1:
            true = true.squeeze(1)
            
        occurrence_logits = pred[:, 0, :, :]
        intensity_pred = pred[:, 1, :, :]
        
        bool_rain = (true > 0).float()
        
        # BCE with logits for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(occurrence_logits, bool_rain, reduction='none')
        
        # Asymmetric L1 or MSE for intensity, computed ONLY where it actually rains
        diff = true - intensity_pred
        intensity_loss = torch.where(diff > 0, self.alpha * torch.abs(diff), torch.abs(diff))
        
        total_loss = bce_loss + (bool_rain * intensity_loss)
        return torch.mean(total_loss)
