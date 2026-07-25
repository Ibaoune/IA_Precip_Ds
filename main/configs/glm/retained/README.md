# Configuration Retenue : GLM (glm_precip_l2)

Cette configuration correspond au modèle GLM (Generalized Linear Model) de référence, régularisé via L2 (Ridge).

## Détails de l'Expérience
- **Nom** : `glm_precip_l2`
- **Variable cible** : `precip`
- **Référence** : `mswep`

## Hyperparamètres Principaux
- **Taux d'apprentissage (LR)** : 1e-3
- **Taille de batch** : 64
- **Époques** : 10
- **Fonction de perte** : Bernoulli-Gamma
- **Weight Decay (Alpha L2)** : 0.01
- **Optimiseur** : AdamW
