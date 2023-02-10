<h1 align="center">MASTER 2 ÉCONOMÉTRIE ET STATISTIQUES APPLIQUÉES</h1>
<h1 align="center">SVM ET RÉSEAUX DE NEURONES</h1>
<h1 align="center">CLASSIFICATION D'IMAGES DE SPORTS</h1>
<p align="center">
    <img width="400" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTCUbX6dp50ELEEyOaKdvFkpucInSMbYoaL2A&usqp=CAU" alt="reseaux">
</p>
<h1 align="center">PAR</h1>
<h1 align="center">ALEXANDRE GLOANNEC</h1>
<h1 align="center">ESTEBAN PASCAL</h1>
<h1 align="center">SOUS LA DIRECTION DE M. BENJAMIN ROUL</h1>
<h1 align="center">FÉVRIER 2023</h1>

# ***Problématique de l'étude de cas***
# Classification d'images de balles de sports

<p align="justify">  Le machine learning est la science du développement d'algorithmes et de modèles statistiques que les systèmes informatiques utilisent pour effectuer des tâches sans instructions explicites, en s'appuyant sur des modèles et des déductions. Les systèmes informatiques utilisent des algorithmes de machine learning pour traiter de grandes quantités de données historiques et identifier des modèles de données. Cela leur permet de prédire les résultats avec davantage de précision à partir d'un jeu de données d'entrée donné. Par exemple, les scientifiques des données pourraient entraîner une application médicale en vue de diagnostiquer le cancer à partir d'images radiographiques en stockant des millions d'images numérisées et les diagnostics correspondants. </p>  

<p align="justify"> L'objectif de ce projet a donc été d'appliquer la méthodologie d'un projet de Machine Learning sur un jeu de données concernant la classification d'images de balles de sports. Plus de 6 000 images de ballons de sport de 15 sports différents compose ce jeu de données. Ce sont : le football américain, le baseball, le basketball, la boule de billard, le bowling, la balle de cricket, le football, la balle de golf, la balle de hockey sur gazon, la rondelle de hockey, le ballon de rugby, le volant, la balle de tennis de table, la balle de tennis et le volley-ball. Le projet en question correspond à un projet issu du site Kaggle. Nous avons utilisé la base Kaggle "Sports balls - multiclass image classification". N'ayant pas encore eu l'occasion de travailler avec une classification d'images, nous avons dès lors eu la volonté de nous challenger afin de réaliser un projet différent de ce dont nous avions l'habitude. Néanmoins, excepté durant le cours d'SVM et réseaux de neurones de cette année, nous n'avions jamais travaillé auparavant sur une base de données composée d'images à classifier. Ce qui nous a alors posé quelques problèmes pour l'importation des données puisque lors de ce cours nous avions pu observer l'exemple concernant la base de données "MNIST", cependant l'importation de cette dernière est bien différente de la nôtre. En effet, trouver une solution exploitable pour importer les images nous a pris un certain temps.  Nous avons donc procédé de deux manières : dans un premier temps, une méthode qui a pu nous permettre de visualiser les images de la base de données, mais avec laquelle nous n'avons pas réussi à modéliser. Dans un second temps, une méthode qui nous a cette fois-ci permis de modéliser, mais avec laquelle nous n'avons pas réussi à visualiser les images. Durant ce projet, nous utiliserons donc une combinaison de ces deux méthodes afin de réaliser un modèle de classification multiclass. </p>    

Source des données : https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification 

# I. Importation du jeu de données

## 1ère méthode : Importer les images afin de les visualiser  

<p align="justify"> Dans un premier temps, nous avons mis en place une méthode d'importation des données afin de visualiser les images présentes au sein du jeu de données. Néanmoins, cette méthode ne nous a pas permis de modéliser quoi que ce soit mais nous a aider plus-tard au cours de l'analyse a observer quelles images nos modèles avaient du mal à prédire. </p>  

<p align="justify"> Nous observons ci-dessous à titre d'exemple quelques images ayant pu être visualisées pour 3 catégories de sports différents de manière aléatoire : </p>

<p align="center"> Figure 1 - Image issue de la catégorie baseball </p>
<p align="center">
    <img width="400" src="https://user-images.githubusercontent.com/118008489/218047372-31b382ea-d828-4000-a840-8f6fadec111d.png" alt="baseball">
</p>

<p align="center"> Source : Gloannec A. & Pascal E. (2023). Classification d'images de balles de sports. Extrait du logiciel Python </p>  

<p align="center"> Figure 2 - Image issue de la catégorie football </p>
<p align="center">
    <img width="400" src="https://user-images.githubusercontent.com/118008489/218063229-4beb54fa-7dd0-4562-87f5-c8b354362234.png" alt="football">
</p>

<p align="center"> Source : Gloannec A. & Pascal E. (2023). Classification d'images de balles de sports. Extrait du logiciel Python </p>

<p align="center"> Figure 3 - Image issue de la catégorie rugby </p>
<p align="center">
    <img width="400" src="https://user-images.githubusercontent.com/118008489/218063576-44bb812e-be12-427d-9c46-0c95c37cf5b0.png" alt="rugby">
</p>

<p align="center"> Source : Gloannec A. & Pascal E. (2023). Classification d'images de balles de sports. Extrait du logiciel Python </p>

<p align="justify"> Bien que cette méthode soit utile afin d'importer des images, elle n'est pas exploitable pour réaliser une classification. En effet, le premier problème de cette méthode est que les images sont importées sous forme d'une array à 4 dimensions, ce qui rend difficile la modélisation. Le deuxième problème est que nous n'avons pas réussi à exploiter de variables explicatives, ce qui nous aurait ainsi permis de réaliser une classification.
Afin de régler ces problémes, nous avons donc procédé d'une autre manière pour importer nos images. Ce qui nous a donc aider à obtenir des variables explicatives et donc de réaliser notre classification. </p>  

## 2ème méthode : Importer les images afin de les modéliser

<p align="justify"> Dans un souci de clarté et de temps de calcul pour la performance de notre code, nous avons été obligé de supprimer deux catégories, à savoir "tennis" et "bowling". Après concertation, ces deux catégories à écarter ont été décidés par nous-même.
À noter que plus-tard au cours de l'analyse, nous procéderons à différentes méthodes de rééchantillonage avec nos catégories conservées. </p>  

<p align="justify"> Nous avons essayé de visualiser les images à l'aide du code issue de la démo SVM présente au sein du chapitre 1 concernant les SVM sous python. Nous avons aussi essayé d'autres méthodes mais en vain. </p>

<p align="justify"> Ci-dessous, au sein du tableau 1, nous pouvons tout de même observer la répartition du nombre d'images au sein de chacune des catégories : </p>

<p align="center"> Tableau 1 - Répartition du nombre d'images au sein de chacune des catégories </p>

<div align="center">
    
| Catgéories           | Images        |
| -------------        | ------------- |
| basketball           | 340           |
| billiard_ball        | 646           |
| cricket_ball         | 581           |
| football             | 604           |
| golf_ball            | 549           |
| hockey_puck          | 690           |
| rugby_ball           | 493           |
| shuttlecock          | 429           |
| table_tennis_ball    | 620           |
| volleyball           | 432           |
    
</div>  

<p align="center"> Source : Gloannec A. & Pascal E. (2023). Classification d'images de balles de sports. </p>

<p align="justify">  La répartition des images présentes au sein de chaque catégorie est sensiblement égale entre chaque catégorie. Ainsi, un rééchantillonage de notre jeu de données n'est a priori pas nécessaire, mais par souci d'analyse approfondie, nos modélisations, qui seront présentées dans la partie suivante, seront à la fois réalisées avec et sans rééchantillonage de notre jeu de données.  </p>  

# II. Data Preparation

## Anayse du jeu de données

*   Analyse des variables (Variable d'intérêt / Variables explicatives)
*   Identification des valeurs manquantes
*   Identification et correction des outliers

## Nettoyage du jeu de données

*   Imputation de valeurs manquantes 

## Standardisation du jeu de données

<p align="justify"> Étant donné notre jeu de données composé d'images à classifier, l'ensemble des étapes énoncées ci-dessous au sein de la partie consacrée à la préparation des données n'ont pas été / pu être réalisées. Le cas échéant, celles-ci auraient du être développées de manière rigoureuse.
C'est pour cette raison que nous sommes directement passé à notre étape de modélisation. </p>  

# III. Séparation de notre jeu de données train

<p align="justify"> La validation croisée est une technique d’apprentissage supervisé pour vérifier la fiabilité d’un modèle. </p>  

<p align="justify"> Ici nous utlisons la validation croisée Hold Out pour séparer notre jeu train en deux nouvelles bases : </p>  
    
<p align="justify"> - Un échantillon d’apprentissage pour entraîner les paramètres du modèle. </p> 
<p align="justify"> - Un échantillon de validation pour vérifier la déviance du modèle. </p> 

<p align="justify"> Le split du jeu de données s'est effectué de la manière suivante : 80% du jeu de données en jeu train et 20% du jeu de données en jeu test. </p>  

# IV. Modélisation

<p align="justify"> Nous allons maintenant présenter la modélisation et les résultats obtenus en termes de performance pour chacun de nos modèles choisis dans cette étude avec nos catégories retenues. À noter que nous présenterons pour chacun de nos modèles à la fois les résultats de nos modélisations obtenus sans rééchantillonage et avec rééchantillonage du jeu de données. </p>  

<p align="justify"> Concernant les méthodes que nous avons employé durant cette étude de cas, voici l'ordre dans lequel celles-ci ont été mises en place : </p>  

## Méthodes employées :
    
   ### - Partie OVR :
    
   * OneVSRestClassifier (SVC)
   * OneVsRestClassifier (LinearSVC)
   * OneVsRestClassifier (SGDClassifier) 
    
   ### - Partie OVO :
    
   * OneVsOneClassifier (SVC)
   * OneVsOneClassifier (LinearSVC)
   * OneVsOneClassifier (SGDClassifier) 

   ### - MLP avec keras
   ### - Arbre de décision
   ### - GradientBoostingClassifier
   ### - Random Forest
   ### - Ridge Classifier
   ### - Régression Logistique Multiclass

<p align="justify"> Et voici la démarche suivie pour cette partie concernant la modélisation : </p>  

   * Application des modèles à l’échantillon test.
   * Comparaison de la performance des modèles (Matrice de confusion, Accuracy).

<p align="justify"> Nous introduirons tout d'abord chaque méthode avant de présenter à la fin de cette partie un tableau récapitulatif résumant pour chacune d'entre elles à la fois, la qualité de prévision du modèle, ainsi que le nombre de fois ou le modèle prédit d'une mauvaise manière les images pour chaque catégorie </p>    

### - Partie OVR 

 * OneVsRestClassifier (SVC)
 * OneVsRestClassifier (LinearSVC)
 * OneVsRestClassifier (SGDClassifier) 

<p align="justify"> Stratégie multiclass/multilabel OneVSRestClassifier (OVR), également connue sous le nom de OneVSAll. Cette approche consiste à ajuster un classifieur par classe. Pour chaque classificateur, la classe est ajustée par rapport à toutes les autres classes. En plus de son efficacité de calcul (seuls les classificateurs n_classes sont nécessaires), l'un des avantages de cette approche est son interprétabilité. Étant donné que chaque classe est représentée par un et un seul classificateur, il est possible d'acquérir des connaissances sur la classe en inspectant son classificateur correspondant. Il s'agit de la stratégie la plus couramment utilisée pour la classification multiclasse et c'est un choix par défaut équitable. </p>  
   
### - Partie OVO :
    
  * OneVsOneClassifier (SVC)
  * OneVsOneClassifier (LinearSVC)
  * OneVsOneClassifier (SGDClassifier) 

<p align="justify"> OneVsOneClassifier construit un classificateur par paire de classes. Au moment de la prédiction, la classe qui a reçu le plus de votes est sélectionnée. En cas d'égalité (entre deux classes avec un nombre égal de votes), il sélectionne la classe avec la confiance de classification agrégée la plus élevée en additionnant les niveaux de confiance de classification par paire calculés par les classificateurs binaires sous-jacents. Puisqu'elle nécessite d'adapter n_classes * (n_classes - 1) / 2 classificateurs, cette méthode est généralement plus lente que OneVSRestClassifier, en raison de sa complexité O (n_classes^2). La fonction de décision est le résultat d'une transformation monotone de la classification un contre un. </p>  

### - MLP avec keras

<p align="justify"> En intelligence artificielle, plus précisément en apprentissage automatique, le perceptron multicouche (multilayer perceptron, MLP) est un type de réseau neuronal artificiel organisé en plusieurs couches. L'information circule de la couche d'entrée vers la couche de sortie uniquement : il s'agit donc d'un réseau à propagation directe (feedforward). Chaque couche est constituée d'un nombre variable de neurones, les neurones de la dernière couche (dite « de sortie ») étant les sorties du système global. </p>  

<p align="justify"> Malheureusement nous n'avons pas réussi à obtenir quelconque résultat interprétable à l'aide de la méthode MLP avec Keras.  
Cependant, nous avons souhaité tester d'autres modèles que ceux vus en cous. Par conséquent, nous avons décidé du choix de ces modèles à l'aide de la documentation suivante : https://scikit-learn.org/stable/modules/multiclass.html </p> 

 ### - Arbre de décision
 
 <p align="justify"> Un arbre de décision est un outil d'aide à la décision représentant un ensemble de choix sous la forme graphique d'un arbre. Les différentes décisions possibles sont situées aux extrémités des branches (les « feuilles » de l'arbre), et sont atteintes en fonction de décisions prises à chaque étape. L'arbre de décision est un outil utilisé dans des domaines variés tels que la sécurité, la fouille de données, la médecine, etc. Il a l'avantage d'être lisible et rapide à exécuter. Il s'agit de plus d'une représentation calculable automatiquement par des algorithmes d'apprentissage supervisé. </p>  
 
### - GradientBoostingClassifier

<p align="justify"> Le gradient boosting est un modèle ensembliste de boosting utilisant la descente de gradient pour optimiser une fonction de perte. Cette descente est utilisée pour le calcul des résidus des individus, lors de la construction du modèle suivant.  
Cette méthode est implémentée dans la librairie scikit-learn : GradientBoostingRegressor et GradientBoostingClassifier. </p>  

<p align="justify"> Au sujet du modèle GradientBoostingClassifier, celui-ci a nécessité un nombre conséquent de temps de calcul rien qu'avec 3 catégories afin de tester le modèle. C'est pourquoi nous n'avons pu malheureusement l'appliquer à l'ensemble de nos catégories et l'avons alors mis de côté. </p>  

### - Random Forest

<p align="justify"> Le Random Forest est un modèle ensembliste basé sur la construction de multiples arbres de décision (CART).
Chaque arbre est construit à partir d’un échantillon aléatoire avec remise des individus et des variables. Lors d’une classification, la prédiction finale est la modalité majoritairement prédite. </p>  
    
<p align="justify"> Les avantages sont les suivants : 
    
 * Accepte les variables qualitatives et quantitatives  
 * Pas besoin de vérifier des hypothèses de normalité et de variance  
 * Il répond aux problèmes de classification et régression  
 * Modèle relativement puissant sur des problèmes de modélisation complexe </p>   

<p align="justify"> Tandis que les inconvénients sont :  
    
 * Difficilement interprétable  
 * Attention au sur-apprentissage </p>   

### - Ridge Classifier

<p align="justify"> En apprentissage automatique, la classification ridge est une technique utilisée pour analyser les modèles discriminants linéaires. C'est une forme de régularisation qui pénalise les coefficients du modèle pour éviter le surajustement. Le surajustement est un problème courant dans l'apprentissage automatique qui se produit lorsqu'un modèle est trop complexe et capture le bruit dans les données au lieu du signal sous-jacent. Cela peut conduire à de mauvaises performances de généralisation sur de nouvelles données. La classification Ridge résout ce problème en ajoutant un terme de pénalité à la fonction de coût qui décourage la complexité. Il en résulte un modèle plus apte à généraliser à de nouvelles données.  </p>  

### - Régression Logistique Multiclass

<p align="justify"> La régression logistique est une technique d'analyse de données qui utilise les mathématiques pour trouver les relations entre deux facteurs de données. Elle utilise ensuite cette relation pour prédire la valeur de l'un de ces facteurs en fonction de l'autre. La régression logistique est une technique importante dans le domaine de l'intelligence artificielle et du machine learning. Les modèles de régression logistique peuvent traiter de grands volumes de données à grande vitesse, car ils nécessitent moins de capacité de calcul, comme la mémoire et la puissance de traitement. </p>   

<p align="justify"> Il est important de savoir qu'afin de réaliser une classification multiclass avec une régression logistique, nous avons précisé au sein des paramètres l'argument multi_class = "multinomial". </p>

<p align="justify"> Pour l'ensemble des méthodes, hormis la régression logistique, les paramètres par défaut des modèles ont été utilisés. D'autre part, nous n'avons pas pu mettre en place les méthodes de GridSearch et RandomozidesGridSearch pour notre modélisation. À nouveau, la problématique du temps de calcul de ces méthodes pour nos données est rentrée en compte. </p>
    
# V. Rééchantillonage des données

<p align="justify"> Précédemment, nous avons vu que certaines catégories étaient malgré tout sous-représentées vis-à-vis de certaines.  Nous allons donc procéder à de l'undersumpling afin d'améliorer potentiellement la qualité de nos modèles. Le sous-échantillonnage implique d'introduire un biais pour sélectionner plus d'échantillons d'une classe que d'une autre, afin de compenser un déséquilibre déjà présent dans les données ou susceptible de se développer si un échantillon purement aléatoire était prélevé. </p>  

<p align="justify"> À noter que nous n'avons pas procédé également à de l'oversampling car cette méthode prenait beaucoup trop de temps de calcul dans le cas de notre analyse. C'est la raison pour laquelle nous avons concentré nos efforts sur l'undersampling. </p>  

<p align="justify">  Voici donc au sein de la figure 4 la nouvelle répartition après Undersampling de nos données : </p>    

<p align="center"> Figure 4 - Nouvelle répartition après Undersampling de nos données </p>
<p align="center">
    <img width="400" src="https://user-images.githubusercontent.com/118008489/218097925-50b38942-c9c1-494f-883a-e25031580641.png" alt="undersampling">
</p>

<p align="center"> Source : Gloannec A. & Pascal E. (2023). Classification d'images de balles de sports. Extrait du logiciel Python </p>

<p align="justify"> Ci-dessous, au sein du tableau 2, nous observons la nouvelle répartition du nombre d'images au sein de chacune des catégories après undersampling: </p>

<p align="center"> Tableau 2 - Répartition du nombre d'images au sein de chacune des catégories après undersampling </p>

<div align="center">
    
| Catgéories           | Images        |
| -------------        | ------------- |
| american_football    | 340           |
| baseball             | 340           |
| basketball           | 340           |
| billiard_ball        | 340           |
| bowling_ball         | 340           |
| cricket_ball         | 340           |
| football             | 340           |
| golf_ball            | 340           |
| hockey_ball          | 340           |
| hockey_puck          | 340           |
| rugby_ball           | 340           |
| shuttlecock          | 340           |
| table_tennis_ball    | 340           |
| volleyball           | 340           |
    
</div>  

<p align="center"> Source : Gloannec A. & Pascal E. (2023). Classification d'images de balles de sports. </p>

<p align="center">  Cette fois-ci le nombre de catégories est supérieure étant donné qu'avec la procédure d'Undersampling nous ne rencontrons plus de problème quant à la richesse de notre base de données. </p>
    
<p align="justify"> Sans oublier que nous séparons dans ce cas ci-présent aussi notre jeu train en deux nouveaux jeux de données pour la modélisation : 80% du jeu de données en jeu train et 20% du jeu de données en jeu test. </p>  

# VI. Interprétation du meilleur modèle

<p align="justify"> Enfin, passons au récapitulatif des résultats en termes de performances de l'ensemble des modèles. Nous représentons au sein du tableau 3 la qualité de prévision de l'ensemble des modèles. Notons que nous ne pouvons comparer les qualités de prévision entre les deux méthodes. Tandis qu'au sein des tableaux 4 et 5, c'est le nombre d'erreur réalisé pour le meilleur modèle, avant et après rééchantillonage, comme image mal prédite pour chaque catégorie que nous observons. Enfin, les tableaux 6 et 7 montrent les matrices de confusion obtenues pour la méthode Random Forest avant et après rééchantillonage. </p>

<p align="center"> Tableau 3 - Qualité de prévision de l'ensemble des modèles </p>

<div align="center">
    
| Modèles                             | Avant rééchantillonage  | Après rééchantillonage |
| -------------                       | -------------           | -------------          |
| OneVSRestClassifier (SVC)           | 49,16%                  | 44,66%                 |
| OneVsRestClassifier (LinearSVC)     | 39,23%                  | 34,14%                 |
| OneVsRestClassifier (SGDClassifier) | 29,79%                  | 20,25%                 |
| OneVSOneClassifier (SVC)            | 41,79%                  | 34,72%                 |    
| OneVsOneClassifier (LinearSVC)      | 41,79%                  | 37,28%                 |
| OneVsOneClassifier (SGDClassifier)  | 31,66%                  | 26,83%                 |
| Arbre de décision                   | 32,35%                  | 29,16%                 |
| Random Forest                       | 51,23%                  | 47,66%                 |
| Ridge Classifier                    | 36,97%                  | 31,94%                 |  
| Régression Logistique Multiclass    | 26,70%                  | 25,95%                 |
    
</div>  

<p align="center"> Source : Gloannec A. & Pascal E. (2023). Classification d'images de balles de sports. </p>

<p align="center"> Tableau 4 - Nombre d'erreur réalisé pour le modèle Random Forest avant rééchantillonage comme image mal prédite pour chaque catégorie </p>

<div align="center">
    
|                  | Random Forest |
| ------------     | ------------- |
| basketball       | 46            |
| billiard_ball    | 39            |
| cricket_ball     | 36            |           
| football         | 69            |
| golf_ball        | 60            |
| hockey_puck      | 43            | 
| rugby_ball       | 50            |
| shuttlecock      | 38            |
| table_tennis_ball| 51            |
| volleyball       | 64            |
    
</div>  

<p align="center"> Source : Gloannec A. & Pascal E. (2023). Classification d'images de balles de sports. </p>

<p align="center"> Tableau 5 - Nombre d'erreur réalisé pour le modèle Random Forest après rééchantillonage comme image mal prédite pour chaque catégorie </p>

<div align="center">
    
|                  | Random Forest |
| ------------     | ------------- |
| american_football| 53            |
| baseball         | 61            |
| basketball       | 32            |
| billiard_ball    | 56            |
| bowling_ball     | 48            |
| cricket_ball     | 40            |
| football         | 61            |
| golf_ball        | 61            |
| hockey_ball      | 52            |
| hockey_puck      | 49            |
| rugby_ball       | 57            |
| shuttlecock      | 33            |
| table_tennis_ball| 54            |
| volleyball       | 65            |
    
</div>  

<p align="center"> Source : Gloannec A. & Pascal E. (2023). Classification d'images de balles de sports. </p>

<p align="center"> Tableau 6 - Matrice de confusion du modèle Random Forest avant rééchantillonage </p>
<p align="center">
    <img width="800" src="https://user-images.githubusercontent.com/118008489/218124162-c0676566-4454-43d2-b6db-e3d4765ad1d9.png" alt="matrice">
</p>

<p align="center"> Source : Gloannec A. & Pascal E. (2023). Classification d'images de balles de sports. Extrait du logiciel Python </p>

<p align="center"> Tableau 7 - Matrice de confusion du modèle Random Forest après rééchantillonage </p>
<p align="center">
    <img width="800" src="https://user-images.githubusercontent.com/118008489/218125518-3d0ad316-23d3-4ff2-92f4-60af366d8146.png" alt="matrice1">
    <img width="400" src="https://user-images.githubusercontent.com/118008489/218125644-c677a28e-ccea-4b08-b619-c13ab8709e8d.png" alt="matrice2">
</p>

<p align="center"> Source : Gloannec A. & Pascal E. (2023). Classification d'images de balles de sports. Extrait du logiciel Python </p>

<p align="justify"> Pour conclure cette analyse concernant la classification d'images de balles de sports, il s'avère que le meilleur modèle obtenu pour classifier ces dernières de manière optimale s'avère être le modèle Random Forest, avant et après rééchantillonage, avec une qualité de prévision, respectivement, de 51,23% et 47,66%. Ce modèle prédit donc correctement les images de balles de sports pour chacune des catégories dans environ 50% des cas. </p>  

<p align="justify"> Au sujet du nombre d'erreur réalisé pour le modèle Random Forest avant rééchantillonage comme image mal prédite pour chaque catégorie, le modèle réalise le minimum d'erreur pour les catégories billiard_ball, cricket_ball et shuttlecock avec, respectivement, 39 erreurs, 36 erreurs et 38 erreurs. À l'inverse, le modèle réalise le maximum d'erreur pour les catégories football et volleyball, avec 69 erreurs pour l'une contre 64 erreurs pour l'autre.</p>  
<p align="justify"> Par ailleurs, pour le nombre d'erreur réalisé pour le modèle Random Forest après rééchantillonage comme image mal prédite pour chaque catégorie, le modèle réalise le minimum d'erreur pour les catégories basketball et shuttlecock, avec 32 erreurs pour l'une contre 33 erreurs pour l'autre. À l'inverse, le modèle réalise le maximum d'erreur pour les catégories baseball, football, golf_ball et volleyball avec 61 erreurs pour les 3 premières contre 65 erreurs pour la dernière.</p>  

<p align="justify"> Pour approfondir les précédentes observations, nous décidons alors d'observer les matrices de confusion du modèle Random Forest avant et après rééchantillonage. </p>   

<p align="justify"> Pour la première matrice de confusion, avant rééchantillonage, certes le modèle prédit la catégorie billiard_ball comme étant l'une d'entre elles avec le plus petit nombre d'erreur d'image mal prédite. Seulement, a contrario des catégories cricket_ball et shuttlecock, elles aussi très performantes, le modèle a tendance à prédire beaucoup d'autres balles de sport comme étant des boules de billard. Cela peut notamment s'expliquer du fait que la boule de billard soit ronde et que notre base de données en contient beaucoup de cette forme, au contraire de forme de balles telles que les balles de cricket et de volant de badminton. En effet, les catégories les moins bien prédites sont le football et le volleyball, ce qui coincide. </p>   

<p align="justify"> Pour la seconde matrice de confusion, après rééchantillonage, le constat semble être le même. Des balles de sport rondes apparaissent comme étant les moins bien prédites, hormis pour les balles de basketball étant très bien prédites. De plus, le modèle se trompe souvent en associant des balles de sport rondes à des boules de billard. </p>   
