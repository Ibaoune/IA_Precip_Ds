# Configuration Retenue : CNN (cnn_exp5)

Cette configuration représente le meilleur compromis trouvé pour le modèle CNN (Convolutional Neural Network) dans la tâche de descente d'échelle.

## Détails de l'Expérience
- **Nom** : `cnn_exp5`
- **Variable cible** : `precip`
- **Référence** : `mswep`

## Hyperparamètres Principaux
- **Taux d'apprentissage (LR)** : 1e-4
- **Taille de batch** : 64
- **Époques** : 200
- **Fonction de perte** : Bernoulli-Gamma (optimale pour les précipitations)
- **Normalisation** : Gridbox
- **Scheduler** : Cosine Annealing
- **Dropout** : 0.2
- **Weight Decay** : 1e-5
