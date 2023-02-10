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


