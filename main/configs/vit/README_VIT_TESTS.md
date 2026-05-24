<!-- Author: M. El Aabaribaoune (@um6p) -->
# Campagne d'Expérimentations et Optimisation du Vision Transformer (ViT) pour la Descente d'Échelle de Précipitations

Ce document répertorie de manière exhaustive les architectures, les corrections théoriques et les 20 configurations d'expérimentation conçues pour évaluer et optimiser le modèle **Vision Transformer (ViT)** sur la tâche de descente d'échelle (downscaling) de précipitations (ERA5 vers MSWEP).

---

## 🛠️ 1. Évolution Architecturale et Corrections Théoriques

### A. Mise à niveau du ViT (`vit_arch.py`)
L'architecture a été entièrement mise à jour pour s'aligner sur les standards les plus performants :
- **Plongements positionnels 2D fixes (`get_2d_sincos_pos_embed`)** : Remplacement des plongements appris par une grille sin/cos 2D, indispensable pour préserver la cohérence géospatiale des grilles climatiques.
- **Interpolation et Sur-échantillonnage flexibles** : Intégration de différents modes d'interpolation (`linear`, `bilinear`, `bicubic`) en amont du Transformer et dans l'Upsampling Decoder.
- **Forçages Saisonniers (Seasonal Forcings)** : Prise en compte optionnelle des vecteurs cos/sin du jour de l'année pour guider le modèle sur les cycles annuels.
- **Couche de Sortie Spécialisée (Bernoulli-Gamma)** : Intégration directe des fonctions d'activation (`sigmoid` pour l'occurrence, `softplus` pour les paramètres de forme $\alpha$ et d'échelle $\beta$).

### B. Résolution de la Sous-estimation des Précipitations
Les analyses mathématiques de la fonction de perte `BernoulliGammaLoss` ($E[X] = \pi \cdot \alpha \cdot \beta$) ont mis en évidence trois freins majeurs, désormais corrigés :
1. **Désactivation du seuil DeepESD (-0.99 mm/jour)** : Dans `preprocessing.py`, la soustraction de $0.99$ mm/jour éliminait les petites pluies et décalait systématiquement l'intensité prédite vers le bas. Le modèle s'entraîne désormais sur les valeurs réelles de précipitations.
2. **Élargissement du Bridage (Clamping)** : Remplacement du `clamp(-5, 5)` par `clamp(-10, 7)` dans le reste du pipeline pour permettre au réseau d'atteindre les valeurs d'échelle nécessaires lors des pics extrêmes.
3. **Normalisation Spatiale** : Migration de `norm_mode: "gridbox"` vers `norm_mode: "channel"` ou `"global"` afin de conserver les gradients spatiaux à l'échelle régionale.

---

## 📊 2. Récapitulatif des 20 Configurations de Test (`configs/vit/tests/`)

Toutes les configurations utilisent la période d'entraînement **1979-2005** et la période de test **2006-2020**.

| Fichier YAML | Nom de l'Expérience | Spécificité / Objectif de l'Expérience | Hyperparamètres Clés |
| :--- | :--- | :--- | :--- |
| **`test_default.yaml`** | `vit_precip_linear` | **Modèle de Référence (Baseline ViT)** | Norm: `channel`, Patch: 4, Dim: 32, Layers: 4, Heads: 2, LR: 1e-3 |
| **`test_exp1.yaml`** | `vit_precip_linear_exp1` | **Validation Baseline corrigée** | Identique au défaut, vérification de l'effet sans retrait de 0.99 mm |
| **`test_exp2.yaml`** | `vit_precip_linear_exp2` | **Capacité de Plongement Supérieure** | Dim latente élargie : `emb_size: 64`, `num_heads: 4` |
| **`test_exp3.yaml`** | `vit_precip_linear_exp3` | **Profondeur Accrue** | Modèle plus profond : `num_layers: 6`, `emb_size: 32` |
| **`test_exp4.yaml`** | `vit_precip_linear_exp4` | **Grands Patchs Spatiaux** | Patchs larges : `patch_size: 8`, `emb_size: 64` |
| **`test_exp5.yaml`** | `vit_precip_linear_exp5` | **Apprentissage Fin et Progressif** | LR réduit : `learning_rate: 5e-4` + Cosine Annealing |
| **`test_exp6.yaml`** | `vit_precip_linear_exp6` | **Régularisation par Dropout** | Lutte contre le surapprentissage : `dropout: 0.2`, `emb_size: 64` |
| **`test_exp7.yaml`** | `vit_precip_linear_exp7` | **Taille de Batch Augmentée** | Stabilité du gradient : `batch_size: 8` |
| **`test_exp8.yaml`** | `vit_precip_linear_exp8` | **Comparaison Optimiseur Adam standard** | Optimiseur : `adam` (sans découplage du weight decay) |
| **`test_exp9.yaml`** | `vit_precip_linear_exp9` | **Comparaison Fonction de Perte MSE** | Perte : `mse` au lieu de `bernoulli_gamma` |
| **`test_exp10.yaml`** | `vit_precip_linear_exp10` | **Modèle Haute Capacité (V1)** | `emb_size: 128`, Layers: 6, Heads: 8, Dropout: 0.15 |
| **`test_exp11.yaml`** | `vit_precip_exp11_global_norm` | **Normalisation Globale & Allongement** | Norm: `global`, `epochs: 10`, `emb_size: 64` |
| **`test_exp12.yaml`** | `vit_precip_exp12_patch2` | **Haute Résolution Spatiale (Patchs Fins)** | Patchs très fins : `patch_size: 2`, `epochs: 10` |
| **`test_exp13.yaml`** | `vit_precip_exp13_bilinear` | **Sur-échantillonnage Bilinéaire** | Interpolation : `bilinear` en amont et décodeur, `epochs: 10` |
| **`test_exp14.yaml`** | `vit_precip_exp14_deep_long` | **Entraînement Long sur Modèle Profond** | `num_layers: 8`, `epochs: 15`, `emb_size: 64` |
| **`test_exp15.yaml`** | `vit_precip_exp15_high_wd` | **Forte Pénalisation L2 (Weight Decay)** | `weight_decay: 1e-3`, `dropout: 0.25`, `epochs: 10` |
| **`test_exp16.yaml`** | `vit_precip_exp16_wide_heads` | **Multi-Têtes d'Attention Élargies** | `num_heads: 8`, `emb_size: 128`, `epochs: 10` |
| **`test_exp17.yaml`** | `vit_precip_exp17_batch16` | **Grand Batch & LR Agressif** | `batch_size: 16`, `learning_rate: 2e-3`, `epochs: 10` |
| **`test_exp18.yaml`** | `vit_precip_exp18_patch2_dim64` | **Combinaison Patchs Fins + Grande Dim** | `patch_size: 2`, `emb_size: 64`, `num_heads: 4`, `epochs: 10` |
| **`test_exp19.yaml`** | `vit_precip_exp19_bicubic` | **Sur-échantillonnage Bicubique + Global** | Interpolation : `bicubic`, Norm: `global`, `epochs: 10` |
| **`test_exp20.yaml`** | `vit_precip_exp20_max_capacity` | **Capacité Maximale du Transformer** | `emb_size: 128`, Layers: 8, Heads: 8, `epochs: 15` |
| **`test_exp21_hybrid_base.yaml`** | `vit_precip_exp21_hybrid_base` | **Hybride 1 : Synthèse de Base (exp11, 14, 15)** | Norm: `global`, Dim: 64, Layers: 8, Epochs: 25, WD: 1e-3 |
| **`test_exp22_hybrid_deep_reg.yaml`** | `vit_precip_exp22_hybrid_deep_reg` | **Hybride 2 : Ultra-Profondeur & Régularisation** | Layers: 10, Epochs: 30, WD: 2e-3, Dropout: 0.25, LR: 5e-4 |
| **`test_exp23_hybrid_bilinear_channel.yaml`**| `vit_precip_exp23_hybrid_bilinear_channel`| **Hybride 3 : Upsampling Bilinéaire & Channel** | Norm: `channel`, Interp: `bilinear`, Layers: 8, Epochs: 25 |
| **`test_exp24_hybrid_wide_global.yaml`** | `vit_precip_exp24_hybrid_wide_global` | **Hybride 4 : Attention Large sous Norm Globale** | Dim: 128, Heads: 8, Layers: 6, Norm: `global`, WD: 1e-3 |
| **`test_exp25_hybrid_fast_cosine.yaml`** | `vit_precip_exp25_hybrid_fast_cosine` | **Hybride 5 : Apprentissage Rapide & Grand Batch**| Batch: 8, LR: 2e-3, Cosine Annealing, Layers: 8, Epochs: 25 |

---

## 🚀 3. Instructions d'Exécution sur SLURM

Les expérimentations sont divisées en lots pour faciliter la gestion des ressources sur la partition GPU :

### Soumission du Lot 1 (Tests 1 à 10)
```bash
cd main/scripts/job_tests/
python submit_all_vit_tests.py
```

### Soumission du Lot 2 (Tests 11 à 20)
```bash
cd main/scripts/job_tests/
python submit_vit_tests_11_to_20.py
```

### Soumission du Lot 3 : Les 5 Configurations Hybrides (Tests 21 à 25)
```bash
cd main/scripts/job_tests/
python submit_vit_hybrids_21_to_25.py
```

Chaque script de soumission alloue de manière autonome 1 nœud GPU avec 32 Go de RAM pour une durée sécurisée de 4 heures par job (`#SBATCH --time=04:00:00`). Les logs de sortie et d'erreur sont enregistrés individuellement sous le format `run_test_expX_%j.log`.
