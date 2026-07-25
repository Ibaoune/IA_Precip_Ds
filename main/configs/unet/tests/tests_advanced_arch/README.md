# Tests d'Architecture Avancée (test_exp1 à test_exp12)

Ce dossier regroupe des expériences (configurations) testant des versions plus complexes et robustes du modèle U-Net :

- **U-Net V2** (Réseau amélioré par défaut avec Dropouts et GroupNorms optionnels)
- **CoordConv U-Net** (Ajout explicite des coordonnées géographiques dans les filtres pour aider le modèle à comprendre la localisation)
- **Attention U-Net** (Intégration de portes d'attention pour se focaliser sur les zones pertinentes lors du décodage spatial)
- **Doury U-Net** (Architecture U-Net personnalisée/optimisée issue des travaux de la littérature/Doury)

Ces expériences visent à trouver la meilleure structure interne pour capturer la complexité spatiale des champs météorologiques de manière optimale, souvent en étudiant également l'application d'un masque de fonction de perte (loss_mask).
