# Retained Baseline CNN Model (CNN_Exp5)

**Author:** M. El Aabaribaoune (@um6p)

This directory contains the final retained configuration (`config.yaml`) for the baseline Convolutional Neural Network (CNN) precipitation downscaling model, internally referred to as `cnn_exp5`.

## Architectural Description

Convolutional Neural Networks (CNNs) have been widely adopted in statistical downscaling to capture complex, non-linear spatiotemporal relationships between large-scale atmospheric predictors and local climate variables (e.g., Vandal et al., 2017; Baño-Medina et al., 2020; Quesada-Chacón et al., 2022). While recent advancements have popularized fully convolutional networks like the U-Net for this task, standard CNN architectures that incorporate fully connected components remain highly relevant due to their unique mapping properties.

The architecture retained here employs a Convolutional-to-Dense topology. It is composed of three sequential convolutional layers utilizing $3 \times 3$ kernels (with 50, 25, and 1 filters respectively, alongside ReLU activations and zero-padding). These layers are designed to extract hierarchical spatial features and synoptic atmospheric patterns from the low-resolution predictor fields. 

Following the feature extraction phase, instead of relying on localized upsampling mechanisms (such as transposed convolutions or bilinear interpolation), the entire spatial field is flattened into a 1D vector. This vector is then passed through a massive fully connected (`Linear`) layer that maps the global low-resolution feature representation directly to the exact dimensions of the high-resolution output grid ($160 \times 170$). 

This global dense mapping constitutes the critical advantage of this architecture: it acts as a domain-wide spatial transformation matrix. By explicitly connecting every region of the input domain to every pixel of the output grid, the network easily breaks translation invariance and learns the precise influence of static boundary conditions, such as the Atlas Mountains' complex topography and the Moroccan coastlines. Consequently, this architecture excels at reproducing the long-term climatological mean and yields an exceptionally low spatial bias across the entire domain. However, due to the nature of the dense projection, it tends to exhibit a smoothing effect on highly localized, high-frequency extreme precipitation events (e.g., R95p) when compared to specialized encoder-decoder architectures.

## Technical Specifications
- **Python Model Class:** `CNN` (defined in `main/src/models/cnn.py`)
- **Convolutional Filters:** [50, 25, 1]
- **Loss Function:** RMSE / Bernoulli-Gamma (probabilistic distribution)
- **Optimization Strategy:** Adam optimizer paired with a Cosine Annealing learning rate scheduler.
- **Regularization:** Dropout ($p=0.2$) and Weight Decay ($10^{-5}$) were rigorously tuned to prevent overfitting on the dense layer.

## Usage
To retrain or evaluate this retained baseline architecture, execute the following commands using this configuration file:
```bash
python3 train.py main/configs/cnn/retained/config.yaml
python3 eval.py main/configs/cnn/retained/config.yaml
```
