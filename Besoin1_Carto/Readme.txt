========================================================================
       OUTIL D'AIDE À LA DÉCISION : GESTION DES ARBRES (IA)
                      VILLE DE SAINT-QUENTIN
========================================================================

Cet outil permet de classer instantanément un arbre dans une catégorie 
de taille (Petit, Moyen ou Grand) en utilisant l'intelligence artificielle.

Les modèles ont été préalablement entraînés sur l'ensemble du patrimoine 
arboré de la ville. Aucune installation technique ou entraînement 
supplémentaire n'est requis de votre part.

------------------------------------------------------------------------
1. CONTENU DU DOSSIER
------------------------------------------------------------------------
- predict_cluster.py   : Le programme à lancer.
- model_kX.pkl         : Cerveau de l'IA (pré-chargé).
- scaler_unique.pkl    : Outil de normalisation (pré-chargé).
- mapping_kX.pkl       : Traducteur des catégories (pré-chargé).

------------------------------------------------------------------------
2. COMMENT LANCER L'OUTIL ?
------------------------------------------------------------------------
Ouvrez votre terminal (invite de commande) dans ce dossier et tapez :

    > python predict_cluster.py

------------------------------------------------------------------------
3. FONCTIONNEMENT DU SCRIPT
------------------------------------------------------------------------
Une fois le script lancé, suivez simplement les instructions à l'écran :

1. CHOIX DE LA PRÉCISION : Tapez '2', '3' ou '4' pour choisir le nombre 
   de catégories de taille souhaitées.
   (Recommandation : Tapez '3' pour obtenir Petit / Moyen / Grand).

2. CARACTÉRISTIQUES : Saisissez la hauteur (en mètres) et le diamètre 
   du tronc (en centimètres).

3. RÉSULTAT : L'IA affiche immédiatement la catégorie correspondante.

------------------------------------------------------------------------
4. EXEMPLE D'UTILISATION
------------------------------------------------------------------------
- Choix catégories : 3
- Hauteur : 18
- Diamètre : 45
> Résultat : L'IA classe cet arbre en catégorie : MOYEN

========================================================================
Projet IA 2026 - Besoin 1