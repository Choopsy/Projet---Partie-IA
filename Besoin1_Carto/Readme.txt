========================================================================
       OUTIL D'AIDE À LA DÉCISION : GESTION DES ARBRES (IA)
                      VILLE DE SAINT-QUENTIN
========================================================================

Cet outil permet de classer instantanément un arbre dans une catégorie 
de taille (Petit, Moyen ou Grand) grâce à un moteur d'Intelligence 
Artificielle pré-entraîné.

------------------------------------------------------------------------
1. DESCRIPTION TECHNIQUE
------------------------------------------------------------------------
L'outil s'appuie sur l'algorithme K-Means (Apprentissage Non Supervisé).
Il utilise deux variables morphologiques pour la prédiction :
- La hauteur totale de l'arbre (mètres).
- Le diamètre du tronc (centimètres).

Le système est livré avec des modèles optimisés pour garantir une 
réponse immédiate sans nécessiter de phase d'entraînement supplémentaire.

------------------------------------------------------------------------
2. CONTENU DU DOSSIER
------------------------------------------------------------------------
- predict_cluster.py   : Programme principal à exécuter.
- export_IA.csv        : Base de données du patrimoine arboré.
- model_kX.pkl         : Fichiers "cerveaux" de l'IA (K=2, 3 et 4).
- scaler_unique.pkl    : Outil de mise à l'échelle des mesures.
- mapping_kX.pkl       : Dictionnaire de traduction (ID -> Texte).
- carte_interactive_besoin1.html : Support visuel de la ville.

------------------------------------------------------------------------
3. MODE D'EMPLOI
------------------------------------------------------------------------
Pour lancer l'application, ouvrez un terminal dans ce dossier et tapez :

    > python predict_cluster.py

Le programme vous guidera ensuite étape par étape :

1. CHOIX DU MODÈLE : Saisissez le nombre de catégories (2, 3 ou 4).
   * Note : La configuration '3' est recommandée pour les services 
     techniques (Petit / Moyen / Grand).

2. SAISIE DES DONNÉES : Entrez la hauteur (m) et le diamètre (cm).

3. RÉSULTAT : L'IA affiche la catégorie de l'arbre et ouvre 
   automatiquement la carte interactive de Saint-Quentin dans votre 
   navigateur pour une analyse visuelle complète.

------------------------------------------------------------------------
4. EXEMPLE D'UTILISATION
------------------------------------------------------------------------
Saisie : 
   - Catégories : 3
   - Hauteur : 15
   - Diamètre : 40
Résultat : 
   - >>> RÉSULTAT : Catégorie **Moyen**
   - [Ouverture de la carte interactive...]

========================================================================
Réalisé dans le cadre du Projet IA - FISA 4 - ISEN
========================================================================