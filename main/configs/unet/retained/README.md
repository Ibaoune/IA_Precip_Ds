# Retained Hybrid U-Net Model (UNet_V5_Parallel / Config32)

**Author:** M. El Aabaribaoune (@um6p)

This directory contains the final retained configuration (`config.yaml`) for the U-Net-based precipitation downscaling model.

## Architectural Description

The U-Net is a convolutional encoder–decoder architecture originally developed for image segmentation (Ronneberger et al., 2015) and subsequently used for climate downscaling and regional climate model emulation (e.g., Quesada-Chacón et al., 2022, 2023; Doury et al., 2023; Lin et al., 2023; Kumar et al., 2022; Ding et al., 2024; Fuentes-Franco et al., 2025). This architecture is composed of two distinct blocks: the encoder and the decoder. The former reduces the input dimension using max pooling layers after a block of convolutions, while the latter enlarges it to a specific dimension using transposed convolutions. 

While this builds upon the standard configuration introduced by Ronneberger et al. (2015) and commonly adopted in climate downscaling applications, we introduced several critical modifications to optimize it for spatial precipitation emulation. First, standard convolutions are translation-invariant, which prevents the network from learning location-specific climate relationships such as proximity to coastlines or complex topography. To resolve this, we implemented a Coordinate Convolution (CoordConv) mechanism (Liu et al., 2018), which breaks this spatial invariance by directly injecting normalized latitude and longitude coordinates as input channels. Second, because downscaling high-resolution fields over large domains strictly limits batch sizes due to memory constraints, we replaced Batch Normalization with Group Normalization (Wu & He, 2018), which computes stable statistics across channel groups independently of batch size. Third, to mitigate the "checkerboard" artifacts frequently induced by transposed convolutions (Odena et al., 2016)—which create unnatural high-frequency noise in precipitation patterns—we utilized bilinear upsampling. Finally, Dropout was integrated for robust regularization.

Furthermore, empirical testing revealed that standard fully convolutional networks (FCNs) struggle to correct domain-wide climatological biases due to their localized receptive fields. To address this fundamental limitation, several architectural configurations were rigorously tested, and this **Parallel Hybrid (Two-Stream) Architecture** (`UNet_Config32`) was ultimately retained as it provided an optimal balance between mean spatial consistency and local extreme capability.

This hybrid architecture is composed of two parallel, independently operating branches that are additively fused at the final layer:
1. **Global Climatology Stream (Dense CNN):** A baseline Convolutional Neural Network that terminates in a massive fully connected (`Linear`) layer, mapping the entire low-resolution input field directly to the high-resolution output grid. This global mapping allows the network to explicitly memorize and correct the mean spatial bias across the entire domain.
2. **Local Extremes Stream (U-Net):** A fully convolutional U-Net structure (encoder–decoder with skip connections) designed to strictly focus on capturing high-frequency spatial variability, localized topographic residuals, and extreme precipitation events.

This dual-stream approach ensures a perfectly unbiased mean precipitation field (inherited from the dense CNN layer) while simultaneously resolving fine-scale intensity peaks (restored by the U-Net).

## Technical Specifications
- **Python Model Class:** `UNet_Config32` (defined in `main/src/models/unet_experiments.py`)
- **Loss Function:** RMSE
- **Probabilistic Framework:** Bernoulli-Gamma setup (when applicable to configuration).
- **Optimization:** Adam optimizer paired with Cosine Annealing, Weight Decay, and Gradient Clipping.

## Usage
To retrain or evaluate this retained architecture, execute the following commands using this configuration file:
```bash
python3 train.py main/configs/unet/retained/config.yaml
python3 eval.py main/configs/unet/retained/config.yaml
```
