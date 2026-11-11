# Projet-1
Projet de comparaison de performance entre un réseau de neurones et des arbres de décisions

La première étape du workflow a consisté à importer les bibliothèques fondamentales de la science des données (numpy, pandas, scikit-learn) et à charger le jeu de données officiel California Housing. L'exploration initiale du DataFrame a révélé une structure matricielle composée de 20 640 observations réparties sur 9 colonnes distinctes. L'analyse descriptive détaillée via df.describe() et df.info() a permis de vérifier l'intégrité des données : toutes les variables sont de type numérique continu (float64), et aucune donnée manquante n'a été détectée (df.isna().sum() égal à zéro sur l'ensemble des colonnes). Les caractéristiques explicatives comprennent le revenu médian du quartier (MedInc), l'âge moyen des maisons (HouseAge), le nombre moyen de pièces (AveRooms), le nombre moyen de chambres (AveBedrms), la population (Population), l'occupation moyenne par logement (AveOccup), ainsi que les coordonnées géographiques (Latitude et Longitude).

L'étape de visualisation des données a permis de comprendre la distribution sous-jacente de chaque variable grâce à la génération d'histogrammes individuels à 30 classes (bins=30). Cette analyse graphique a mis en évidence des caractéristiques essentielles du jeu de données : une asymétrie à droite marquée pour certaines variables quantitatives telles que Population, AveRooms et AveOccup, causée par la présence de valeurs extrêmes. De plus, l'analyse de la variable cible MedHouseVal a révélé un plafonnement artificiel à la valeur maximale de 5,0 (correspondant à 500 000 $), caractérisant une censure supérieure des données d'origine. La distribution géographique de la latitude et de la longitude reflète quant à elle fidèlement la répartition spatiale de la population californienne le long des côtes et dans les grands centres urbains.

Afin de garantir une évaluation rigoureuse et non biaisée du modèle, la matrice des caractéristiques $X$ a été séparée de la variable cible $y$. Le jeu de données a ensuite été divisé en trois sous-ensembles étanches grâce à deux appels successifs à train_test_split :Ensemble d'entraînement (Train Set) : 14 912 échantillons (70 % du dataset total).Ensemble de validation (Validation Set) : 2 632 échantillons (15 % du dataset total).Ensemble de test (Test Set) : 3 096 échantillons (15 % du dataset total).Pour harmoniser les échelles très disparates des différentes variables (par exemple entre MedInc s'étendant de 0,5 à 15 et Population dépassant 35 000), un prétraitement par standardisation Z-score (StandardScaler) a été appliqué. Afin d'éviter strictement toute fuite de données (data leakage), le scaler a été ajusté exclusivement sur le jeu d'entraînement (fit_transform), puis appliqué sans réajustement sur les jeux de validation et de test (transform).

La modélisation a été réalisée en construisant un réseau de neurones artificiels séquentiel à couches denses avec l'écosystème TensorFlow/Keras :Couche d'entrée : dimensionnée pour recevoir les 8 caractéristiques standardisées de $X$.Première couche cachée : 64 neurones fully-connected avec fonction d'activation ReLU (576 paramètres).Deuxième couche cachée : 32 neurones fully-connected avec fonction d'activation ReLU (2 080 paramètres).Couche de sortie : 1 neurone linéaire pour la prédiction continue de la valeur immobilière (33 paramètres).Le modèle comprend au total 2 689 paramètres tous entraînables. La compilation s'est appuyée sur l'optimiseur adaptatif Adam, la fonction de perte de l'erreur quadratique moyenne (mean_squared_error) et la métrique de l'erreur absolue moyenne (mae). L'entraînement a été configuré sur 100 époques avec une taille de lot (batch size) de 32, associé au mécanisme d'arrêt précoce (EarlyStopping) réglé sur une patience de 10 époques par rapport à la perte de validation (val_loss). L'apprentissage s'est interrompu à la 55ᵉ époque, la meilleure performance sur le jeu de validation ayant été atteinte à l'époque 45 avec une val_loss de 0,2682 et restaurée automatiquement.

L'évaluation finale exécutée sur le jeu de test indépendant (3 096 observations jamais vues lors de l'apprentissage) a fourni des résultats très satisfaisants :Erreur Quadratique Moyenne (MSE) : 0,2722.Erreur Absolue Moyenne (MAE) : 0,3528, ce qui signifie qu'en moyenne, les prédictions du réseau de neurones s'écartent de seulement ~35 280 de la valeur réelle des biens.Coefficient de Détermination ($R^2$) : 0,7923.Ce score $R^2$ de près de 0,80 confirme que le réseau de neurones parvient à capturer près de 80 % de la variance totale de la valeur des logements en Californie à partir des 8 variables explicatives fournies. Ce projet démontre l'efficacité d'un perceptron multicouche (MLP) bien structuré et régularisé par Early Stopping pour résoudre des problèmes complexes de régression non linéaire en ingénierie de données immobilières

Afin d'améliorer les performances du premier réseau, une version plus profonde a été construite : trois couches cachées de 128, 64 et 32 neurones (ReLU) suivies d'une sortie linéaire, soit 11 521 paramètres entraînables contre 2 689 pour le premier modèle. L'optimiseur Adam a été conservé, avec un taux d'apprentissage initial de 0,001. Deux callbacks ont été utilisés :

- **EarlyStopping**, avec une patience portée à 15 époques et restauration des meilleurs poids ;
- **ReduceLROnPlateau**, qui divise le taux d'apprentissage par 2 après 5 époques sans amélioration de la val_loss. Il a été déclenché 7 fois, ce qui a ramené le taux de 0,001 à son minimum de 1e-5.

L'entraînement s'est arrêté à la 79ᵉ époque, la meilleure val_loss (0,2599) ayant été atteinte à l'époque 64. Sur le jeu de test, le modèle améliore légèrement le premier réseau :

| Modèle | MSE | MAE | R² |
|---|---|---|---|
| MLP initial (64-32) | 0,2722 | 0,3528 | 0,7923 |
| MLP amélioré (128-64-32) | 0,2612 | 0,3435 | 0,8007 |

Le MSE baisse d'environ 4 % et l'erreur moyenne passe de ~35 280 $ à ~34 350 $. Le gain reste modeste : augmenter la capacité du réseau ne suffit pas à dépasser nettement un R² de 0,80. Les courbes d'apprentissage montrent aussi un écart entre la perte d'entraînement (0,199) et celle de validation (0,260). Il traduit un début de surapprentissage, que l'arrêt précoce et la réduction du taux d'apprentissage permettent de contenir.

Deux modèles à base d'arbres ont ensuite été testés comme points de comparaison. Ils ont été entraînés sur les données non standardisées, car ils ne dépendent pas de l'échelle des variables.

- **Random Forest** (300 arbres, profondeur illimitée) : MSE = 0,2590, MAE = 0,3314, R² = 0,8024. Il atteint un niveau équivalent au MLP amélioré, sans réglage particulier.
- **XGBoost** (600 arbres, learning_rate = 0,03, max_depth = 8) : MSE = 0,1955, MAE = 0,2866, R² = 0,8508. C'est de loin le meilleur modèle.

## Conclusion

| Modèle | MSE | MAE | MAE en $ | R² |
|---|---|---|---|---|
| MLP initial | 0,2722 | 0,3528 | ~35 280 | 0,7923 |
| MLP amélioré | 0,2612 | 0,3435 | ~34 350 | 0,8007 |
| Random Forest | 0,2590 | 0,3314 | ~33 140 | 0,8024 |
| **XGBoost** | **0,1955** | **0,2866** | **~28 660** | **0,8508** |

Ce projet avait pour but de prédire la valeur médiane des logements californiens à partir de 8 variables. Les données étaient propres (20 640 observations, aucune valeur manquante). Le pipeline repose sur un découpage Train/Validation/Test (70/15/15) et sur une standardisation ajustée uniquement sur le jeu d'entraînement, pour éviter toute fuite de données.

Les quatre modèles obtiennent un R² compris entre 0,79 et 0,85. Le réseau de neurones capture bien les relations non linéaires, et l'amélioration de son architecture a apporté un petit gain. Les méthodes d'ensemble à base d'arbres sont toutefois plus adaptées à ce type de données tabulaires : XGBoost réduit le MSE d'environ 21 % par rapport au MLP amélioré (et de 25 % par rapport au MLP initial) et ramène l'erreur moyenne à environ 28660$. Il est donc retenu comme meilleur modèle.

Plusieurs pistes d'amélioration sont possibles :

- ajouter des variables dérivées (par exemple le ratio chambres/pièces, ou la distance aux grandes villes à partir de la latitude et de la longitude) ;
- optimiser les hyperparamètres de XGBoost par recherche sur grille ou validation croisée ;
- régulariser le MLP (Dropout, L2, Batch Normalization).
