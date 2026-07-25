# Tests CNN

Ce dossier contient les configurations exploratoires testées pour le modèle CNN avant de s'arrêter sur la configuration `cnn_exp5`.

## Tests Documentés

### `cnn_exp3.yaml`
- **Taux d'apprentissage** : 1e-3 (plus agressif)
- **Taille de batch** : 512 (très large)
- **Dropout** : 0.3 (plus de régularisation)
- **Weight Decay** : 1e-4

Les autres configurations testées (notamment celles pour Gridbox, LR Scheduler, etc.) incluent l'utilisation de différents schedulers, l'activation/désactivation de certaines techniques de régularisation, et des optimisations de la taille de batch.
