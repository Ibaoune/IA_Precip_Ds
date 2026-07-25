# Configuration Retenue : UNet (unet_exp4_coordconv_mse)

Cette configuration représente l'architecture UNet optimisée, enrichie par des CoordConvs. Elle est optimisée principalement pour minimiser l'erreur absolue.

## Détails de l'Expérience
- **Nom** : `unet_exp4_coordconv_mse`
- **Type de modèle** : `unet_coordconv`
- **Variable cible** : `precip`

## Hyperparamètres Principaux
- **Taux d'apprentissage (LR)** : 1e-3
- **Taille de batch** : 64
- **Époques** : 150
- **Fonction de perte** : MSE (Mean Squared Error)
- **Normalisation** : Gridbox
- **Gradient Clipping** : 1.0
- **Group Norm** : Activée (32 groupes)

*Note: La fonction de perte MSE offre la meilleure RMSE, bien que d'autres configurations de tests (comme l'Expérience 5 avec Bernoulli-Gamma documentée dans le dossier tests) sont préférées pour la justesse physique des jours secs.*
