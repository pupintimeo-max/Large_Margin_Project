# Large_Margin_Project
Projet sur les réseaux de neurones utilisant une fonction de perte à large marge

Ce projet s'inspire de l'article de Elsayed et al. (2018) qui démontre l'intérêt d'appliquer une fonction de perte intégrant la notion de large marge à des réseaux de neurones pour la classification d'image. Ce projet explore les expériences et résultats de cet article. De plus, il propose une extension de la méthode à la classification de texte en s'inspirant de l'article de Chattopadhyay et al. (2024).

### Eléments du projet  

mnist : jeux de données MNIST utilisés pour l'entraînement et la validation, contenant notamment les jeux de données d'entrainement bruités à 40%.  

models : modèles de classification  

data_provider.py : fichier contenant la classe générant le dataset MNIST au bon format pour l'entrainement et la validation  

mnist_config.py : fichier de configuration des modèles de classification pour MNIST  

mnist_network.py : fichier contenant la classe définissant le modèle de classification pour MNIST  

text_network.py : fichier contenant la classe définissant le modèle de classification pour IMDB  

demo.ipynb : demo du projet

### Démo  

La démo du projet se présente sous la forme du notebook demo.ipynb. Il présente les principaux résultats obtenus en terme de précision avec les modèles entraînés pour la classification d'image ainsi que les résultats de l'étude de la méthode pour la classification de texte sur le dataset IMDB.

### Modèles :  

Modèle 0 lm : modèle entraîné avec la fonction de perte à large marge avec un dataset d'entraînement normal

Modèle 0 xent : modèle entraîné avec la fonction de perte entropie croisée avec un dataset d'entraînement normal

Modèle 1 : modèle entraîné avec la fonction de perte entropie croisée avec un dataset d'entraînement bruité à 40%

Modèle 2 : modèle entraîné avec la fonction de perte à large marge avec un dataset d'entraînement bruité à 40%

Modèle 3 : modèle entraîné avec la fonction de perte entropie croisée avec un dataset d'entraînement réduit à 68 éléments

Modèle 4 : modèle entraîné avec la fonction de perte à large marge avec un dataset d'entraînement réduit à 68 éléments

Modèle 5 : modèle entropie croisée pour la classification de texte

Modèle 6 : modèle large marge pour la classification de texte

Modèle 7 : modèle combinant entropie croisée et large marge pour la classification de texte

### Bibliographie :  

Elsayed, G., Krishnan, D., Mobahi, H., Regan, K., & Bengio, S. (2018). Large Margin Deep Networks for Classification. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cottle, & R. Garnett (Eds.), Advances in Neural Information Processing Systems (NeurIPS) (Vol. 31). Curran Associates, Inc. https://proceedings.neurips.cc/paper/2018/file/42998cf32d552343bc8e460416382dca-Paper.pdf  

Chattopadhyay, N., Goswami, A., Chattopadhyay A. (2024). Adversarial Attacks and Dimensionality in Text
Classifiers, https://arxiv.org/pdf/2404.02660
