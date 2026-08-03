# Retained Baseline GLM Model (glm_precip_l2)

**Author:** M. El Aabaribaoune (@um6p)

This directory contains the final retained configuration (`config.yaml`) for the Generalized Linear Model (GLM) baseline, internally referred to as `glm_precip_l2`.

## Architectural Description

Generalized Linear Models (GLMs) serve as the fundamental benchmark in statistical climate downscaling. They offer a highly interpretable, robust, and computationally efficient linear mapping between large-scale atmospheric predictors and local high-resolution predictands (e.g., Gutiérrez et al., 2013; Maraun et al., 2010). While deep learning architectures (CNNs, U-Nets, and Transformers) can capture complex non-linear spatial dependencies, a properly regularized GLM remains a critical baseline to quantify the added value of non-linearity in resolving regional climate signals.

The architecture retained here is a regularized grid-to-grid linear regression model. Rather than employing complex spatial feature extraction, it linearly maps the predictor variables directly to the predictand space. To mitigate overfitting and manage multicollinearity among adjacent grid cells and predictor variables, an L2 regularization (Ridge penalty) framework was strictly enforced. This penalizes excessively large coefficients, ensuring that the model generalizes well to unseen climate conditions without artificially amplifying noise in the predictor fields.

Because daily precipitation distribution exhibits highly skewed, zero-inflated characteristics that violate the normality assumptions of standard ordinary least squares (OLS) regression, this GLM was specifically optimized using a **Bernoulli-Gamma** probabilistic framework. This dual-distribution approach separately models the probability of precipitation occurrence (Bernoulli) and the intensity of precipitation amounts (Gamma), making it physically consistent with the stochastic nature of rainfall.

## Technical Specifications
- **Python Model Class:** `GLM` (defined in `main/src/models/glm.py`)
- **Loss Function:** Bernoulli-Gamma (probabilistic distribution)
- **Regularization:** L2 Ridge penalty ($\alpha = 0.01$)
- **Optimization Strategy:** AdamW optimizer with a learning rate of $10^{-3}$, ensuring stable convergence on the linear weights.

## Usage
To retrain or evaluate this retained baseline architecture, execute the following commands using this configuration file:
```bash
python3 train.py main/configs/glm/retained/config.yaml
python3 eval.py main/configs/glm/retained/config.yaml
```
