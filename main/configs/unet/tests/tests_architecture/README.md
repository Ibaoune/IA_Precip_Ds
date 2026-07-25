# Tests d'Architecture de Base (unet_exp1 à unet_exp10)

Ce dossier regroupe les 10 expériences testant différentes architectures fondamentales pour le downscaling :

- **unet_exp1** : Réseau Fully Convolutional (FCN) simple (3 couches)
- **unet_exp2** : FCN avec ajout de CoordConv (injection des coordonnées spatiales)
- **unet_exp3** : FCN de type goulot d'étranglement (AutoEncoder sans skip connections)
- **unet_exp4** : AutoEncoder avec 1 Skip Connection globale (Réseau résiduel)
- **unet_exp5** : Shallow U-Net (1 niveau de descente/montée avec concaténation)
- **unet_exp6** : Shallow U-Net + CoordConv
- **unet_exp7** : U-Net standard classique (2 niveaux)
- **unet_exp8** : U-Net standard (3 niveaux) - Équivalent à la version UNet_V2
- **unet_exp9** : UNet V2 combiné avec CoordConv
- **unet_exp10** : U-Net Hybride Dense (Combinaison d'extracteur U-Net et de projection Dense type CNN)

Ces fichiers de configuration permettent d'évaluer l'impact direct de la structure convolutive et des connexions résiduelles sur les performances du modèle face à un CNN classique (référence).
