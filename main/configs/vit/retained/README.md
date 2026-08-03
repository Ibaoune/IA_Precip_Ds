# Retained Vision Transformer Model (vit_precip_exp21_best_hybrid)

**Author:** M. El Aabaribaoune (@um6p)

This directory contains the final retained configuration (`config.yaml`) for the Vision Transformer (ViT) precipitation downscaling model, internally designated as `vit_precip_exp21_best_hybrid`. This configuration represents the optimal synthesis derived from an extensive hyperparameter search (comprising over 20 configurations) aimed at adapting the Transformer architecture for high-resolution spatial climate emulation.

## Architectural Description

The Vision Transformer (ViT), originally introduced for image classification by Dosovitskiy et al. (2020), completely discards the localized convolutions that form the backbone of traditional spatial models (CNNs and U-Nets). Instead, it adapts the self-attention mechanism—initially designed for natural language processing (Vaswani et al., 2017)—to process 2D spatial fields.

In this architecture, the low-resolution predictor field (ERA5) is divided into non-overlapping patches ($4 \times 4$ pixels). Each patch is linearly projected into a dense 1D embedding vector of size 64. Because the self-attention mechanism is inherently permutation-invariant, a learnable positional encoding is added to these embeddings to retain spatial context. The sequence of patch embeddings is then processed through a deep stack of 8 Transformer blocks. Each block consists of a Multi-Head Self-Attention (MHSA) layer featuring 4 distinct attention heads, followed by a Multi-Layer Perceptron (MLP).

The fundamental advantage of the ViT in climate downscaling lies in its global receptive field. Unlike CNNs, which must stack numerous layers to gradually expand their view of the domain, the self-attention mechanism allows every spatial patch to directly attend to every other patch across the entire grid from the very first layer. This theoretical capability makes Transformers exceptionally well-suited for modeling long-range atmospheric teleconnections and complex synoptic-scale interactions that govern regional precipitation patterns.

However, because Transformers lack the rigid inductive bias of translation invariance found in CNNs, they are highly susceptible to overfitting when trained on limited climate datasets. To stabilize the training and prevent the model from memorizing noise, this optimal configuration incorporates a robust regularization strategy. Specifically, global normalization was applied to preserve the underlying thermodynamic properties of the variables, while Dropout ($p=0.2$) and aggressive Weight Decay ($10^{-3}$) were implemented to constrain the self-attention weights.

## Technical Specifications
- **Python Model Class:** `ViT` (defined in `main/src/models/vit_arch.py`)
- **Embedding Dimension (`emb_size`):** 64
- **Patch Size:** 4
- **Transformer Blocks (`num_layers`):** 8
- **Attention Heads (`num_heads`):** 4
- **Loss Function:** RMSE / Bernoulli-Gamma (probabilistic distribution)
- **Optimization Strategy:** Adam optimizer paired with a Cosine Annealing learning rate scheduler.

## Usage
To retrain or evaluate this retained Transformer architecture, execute the following commands using this configuration file:
```bash
python3 train.py main/configs/vit/retained/config.yaml
python3 eval.py main/configs/vit/retained/config.yaml
```
