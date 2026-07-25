# Tests des Hyperparamètres (Expériences Gridbox)

Ce dossier regroupe toutes les expériences d'optimisation sous forme de grille (Grid Search) appliquées au modèle U-Net :

- **Learning Rate (LR) & Schedulers** (`gridbox_LR_scheduler.yaml`, `gridbox_scheduler.yaml`) : Impact du taux d'apprentissage initial et des stratégies de décroissance (Cosine Annealing, ReduceLROnPlateau).
- **Régularisation (Dropout & Weight Decay)** (`gridbox_dropout.yaml`, `gridbox_weight_decay.yaml`) : Évaluation de l'efficacité de l'extinction aléatoire de neurones et de la pénalisation des poids pour prévenir le surapprentissage.
- **Normalisation** (`gridbox_group_norm.yaml`) : Utilisation du Group Normalization comme alternative au BatchNorm.
- **Stabilisation des Gradients** (`gridbox_gradient_clipping.yaml`) : Écrêtage des gradients pour stabiliser l'apprentissage.
- **Durée d'entraînement** (`gridbox_epochs.yaml`) : Étude de convergence.
- **Combinaison Optimale** (`gridbox_best_combined.yaml`) : Essai regroupant les meilleurs hyperparamètres trouvés au travers des études précédentes.

L'objectif de ces configurations est de définir le cadre de convergence optimal pour l'entraînement de U-Net sur nos données.
