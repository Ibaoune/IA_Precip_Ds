# Tests GLM

Ce dossier documente les configurations testées pour le modèle linéaire (GLM).

## Configurations Expérimentales
- **`alpha_l1.yaml`** / **`alpha_l2.yaml`** : Tests pour comparer les effets des régularisations Lasso (L1) et Ridge (L2) sur le GLM.
- **`interp_bilinear.yaml`** / **`interp_nearest.yaml`** : Impact du mode d'interpolation spatiale initiale des variables prédictrices (bilinéaire vs plus proche voisin).
- **`config_old.yaml`** : Ancienne itération des hyperparamètres (souvent basés sur l'erreur quadratique moyenne - MSE, ou sans régularisation spécifique).
