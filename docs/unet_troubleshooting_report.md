# U-Net Troubleshooting & Failure Analysis Report

## 1. The Core Issue: Inference Collapse on GCMs
During the evaluation phase on ERA5, hybrid U-Net architectures (like `unet_exp32_parallel`) produced metrics that closely rivaled the CNN. However, when these models were transferred to LMDZ data for inference, they suffered a catastrophic collapse, predicting a near-constant ~0.5 mm/day everywhere.

This collapse is caused by two compounding factors: a mathematical broadcasting error in the architecture and a spatial covariate shift induced by gridbox standardization.

## 2. Factor A: The Mathematical Broadcasting Bug
In the `unet_exp32_parallel` architecture, the final prediction was computed by summing the CNN branch and the U-Net branch:
```python
return cnn_out + unet_out
```
- **CNN Output:** 1 channel (Precipitation Logit)
- **U-Net Output:** 3 channels (Occurrence, Log-Shape, Log-Scale) for the Bernoulli-Gamma distribution.

Due to PyTorch's automatic broadcasting, the single CNN value was added to all three Gamma parameters simultaneously. During training on ERA5, the neural network managed to optimize its weights to compensate for this mathematical flaw, producing acceptable results. However, this created an **exponential instability**. 

During inference, the Bernoulli-Gamma loss converts the logits into physical precipitation using exponential functions:
```python
Precipitation = sigmoid(Occ + CNN) * exp(Shape + CNN) * exp(Scale + CNN)
```
Any slight deviation in the CNN prediction on LMDZ data is exponentially magnified, causing the model to either explode (e.g., predicting 152,000 mm/day on certain ERA5 pixels) or completely collapse to zero (0.5 mm/day on LMDZ).

## 3. Factor B: Spatial Covariate Shift (Gridbox vs. Global)
Even a mathematically sound U-Net (like `unet_exp21`) would struggle on LMDZ if trained with `norm_mode: gridbox`.
- **During Training:** Predictors are standardized pixel-by-pixel using ERA5 statistics. This removes stationary topographical features, leaving smooth weather anomalies. The U-Net's spatial convolutions learn to process these smooth maps.
- **During Inference:** LMDZ predictors are standardized using ERA5 statistics. Because LMDZ and ERA5 have different physical resolutions and topographies, the pixel-by-pixel subtraction injects severe, static, high-frequency spatial noise into the predictor maps.
- **The Result:** The CNN (which uses 1x1 convolutions) ignores this spatial noise and performs well. The U-Net (which uses 3x3 convolutions) is highly sensitive to spatial patterns. It perceives this artificial noise as out-of-distribution data and fails to predict extremes.

## 4. The Solution: Global Standardization
To unleash the U-Net's true potential and guarantee stability during inference:
1. **Architecture:** Revert to a mathematically robust U-Net that predicts the 3 Gamma parameters directly (e.g., `unet_exp21`).
2. **Standardization:** Change `norm_mode` from `gridbox` to `global`. This ensures that predictors are standardized using a single domain-wide mean and variance. The spatial coherence of the GCM predictors is preserved, and the U-Net convolutions will no longer be disrupted by artificial gridbox noise.
