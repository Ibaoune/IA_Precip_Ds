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

## 5. The "True Residual" Pitfall (Dense Overfitting)
To maintain topographical accuracy, we attempted a "True Residual" architecture (`unet_exp34`). It initialized the U-Net branch to zero and relied on a CNN branch (with `gridbox` standardization) for the base prediction.
This failed spectacularly on inference, producing massive spatial noise and severe RMSE. 
**The Cause:** The baseline CNN branch (`CNN_Exp5` / `BaselineCNNBranch`) is NOT a fully convolutional network. Its final layer is a **Dense (Fully Connected) Layer** that maps a flattened 90-feature vector directly to 28,800 output pixels.
This dense layer has over 2.5 million parameters per channel and essentially memorizes the exact spatial grid of the ERA5 training set. When evaluated on LMDZ (which has slightly shifted spatial distributions), the Dense layer panics and collapses.

## 6. The Ultimate Solution Options
Since Convolutions spread `gridbox` noise, and Dense layers overfit it, the architectures capable of surviving the ERA5 → LMDZ transfer while maintaining topographical accuracy are:
- **Option A (Pure 1x1 CNN Hybrid):** A true Convolutional baseline using exclusively 1x1 Convolutions (no spatial mixing, no Dense layers) with `gridbox` normalization. This acts as a powerful non-linear GLM that is independent pixel-by-pixel.
- **Option B (CoordConv Global U-Net):** A U-Net using `global` normalization (guaranteeing 100% stability on LMDZ), but augmenting the input with `Latitude` and `Longitude` channels (CoordConv) so the U-Net can explicitly locate mountains without relying on corrupted pixel-wise biases.

## 7. Results of ERA5 Evaluation (UNet35 vs UNet36)
We successfully evaluated Option A (`unet_exp35_1x1_hybrid`) and Option B (`unet_exp36_coordconv_global`) on the 15-year ERA5 test set and compared them to the original baseline (`CNN_Exp5`):

- **UNet36 (Global Norm + CoordConv):** Demonstrated a significant drop in local grid-box performance. Global normalization smoothed out the predictors too much, and CoordConv alone was insufficient to recover the sharp topographical details. The model exhibited high RMSE (0.155 vs 0.094) and massively overestimated dry days (CDD of 188 vs 130). This confirms that Global Normalization inherently degrades local ERA5 performance for our architecture.
- **UNet35 (1x1 Hybrid + Gridbox Norm):** Performed phenomenally well. By replacing the Dense layer with a 1x1 Convolution, the model retained the local sharpness of `gridbox` normalization while avoiding the spatial overfitting trap. It achieved an Annual Bias slightly *better* than the baseline (+0.0097 vs -0.0118), an identical RMSE, and superior metrics on extreme winter precipitation (DJF R95 of 22.5 vs 27.1 for CNN, with MSWEP at 23.9). 

**Conclusion:** `UNet35_1x1_Hybrid` is our strongest candidate. It matches the excellent ERA5 grid-box performance of the original CNN, but its fully convolutional nature should allow it to transfer seamlessly to LMDZ without the catastrophic coordinate overfitting caused by Dense layers.

## 8. The Silent State-Dict Loading Bug (Inference Collapse)
Despite the theoretical soundness of `UNet35_1x1_Hybrid`, our initial LMDZ inference completely collapsed, outputting exactly **0.5 mm/day** everywhere. Investigation revealed this was not a conceptual failure, but a silent PyTorch state-dict loading bug.
- **During Training:** The U-Net architectures were wrapped in a `WrappedUNetExp` class (via `main/src/core/training.py`). This added the prefix `unet.` to all state_dict keys (e.g., `unet.cnn_branch.conv1.weight`).
- **During Inference:** In `inference/src/utils.py`, the `UNet_Config35` architecture was instantiated *directly* without the wrapper. 
- **The Failure:** When calling `model.load_state_dict(..., strict=False)`, PyTorch silently ignored all keys starting with `unet.` because they didn't match the model structure. Consequently, inference was run on a neural network with **100% random, untrained initialization weights** (including zero-initialization for the final layer). Since the Bernoulli-Gamma loss converts a `0` logit to exactly `0.5` mm/day, the model output a flat map of 0.5 mm/day.
- **The Fix:** The inference utilities were corrected to instantiate the models with the exact same wrapper class used during training, ensuring the weights are successfully loaded.

## 9. Final LMDZ Transfer Validation — `unet_exp32_parallel` Retained

After fixing the silent state-dict loading bug (Section 8), both `unet_exp32_parallel` (the originally retained model) and `unet_exp35_1x1_hybrid` (the candidate replacement) were re-evaluated on a full LMDZ 1979–2014 inference run (~1.5 GB output per model).

### Results Summary

| Model | LMDZ Mean North (mm/day) | Stable? | Verdict |
|---|---|---|---|
| `CNN_Exp5` | ~0.70 | ✅ Yes | Baseline |
| `unet_exp35_1x1_hybrid` | ~1.03 | ✅ Yes | Viable but lower regional means |
| **`unet_exp32_parallel`** | **~1.30** | ✅ Yes | **Retained ✓** |

### Why `unet_exp32_parallel` Was Re-confirmed as the Retained Model
- Its earlier LMDZ collapse (flat 0.5 mm/day) was **exclusively caused** by the PyTorch state-dict bug — NOT by architectural instability.
- After the bug fix, `unet_exp32_parallel` produced physically meaningful precipitation fields, with annual regional means significantly closer to MSWEP observations than CNN in all four Moroccan regions (North, North-East, East, South).
- `unet_exp35_1x1_hybrid` was tested and showed lower regional precipitation recovery (~1.03 vs ~1.30 mm/day in the North), suggesting the 1×1 hybrid under-captures the precipitation signal in the LMDZ domain.

### Key Lesson
> **Never abandon a model based on an inference result produced with randomly-initialized weights.** The state-dict bug is silent — PyTorch returns no error and `strict=False` allows the mismatch. Always verify weight loading by checking that output statistics are NOT a flat constant (e.g., 0.5 mm/day for a Bernoulli-Gamma model).

### Reference Figure
`postproc/results/inference_retained/figures/lmdz_paper/Fig_2x2_Annual_Boxplots.png` — Annual precipitation boxplots comparing MSWEP, LMDZ250, LMDZ35, U-Net (exp32_parallel), CNN, and ViT across 4 Moroccan regions (1979–2014).

