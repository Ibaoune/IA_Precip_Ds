# Configuration Retenue : ViT (vit_precip_exp21_best_hybrid)

Cette configuration est la synthèse optimale (modèle "Hybride") issue de la vaste campagne de 20+ tests sur l'architecture Vision Transformer (ViT).

## Détails de l'Expérience
- **Nom** : `vit_precip_exp21_best_hybrid`
- **Type de modèle** : `vit`
- **Variable cible** : `precip`

## Architecture ViT
- **Taille d'intégration (emb_size)** : 64
- **Taille des patchs** : 4
- **Profondeur (num_layers)** : 8
- **Têtes d'attention (num_heads)** : 4

## Hyperparamètres Principaux
- **Taux d'apprentissage (LR)** : 1e-3
- **Taille de batch** : 4
- **Époques** : 25
- **Fonction de perte** : Bernoulli-Gamma
- **Normalisation** : Globale (préserve la thermodynamique)
- **Dropout** : 0.2
- **Weight Decay** : 1e-3
- **Scheduler** : Cosine Annealing
