import pickle
import numpy as np
import time

def main():
    print("====================================================")
    print("   OUTIL DE PRÉDICTION DE TAILLE D'ARBRE (IA)")
    print("====================================================\n")

    try:
        # Chargement obligatoire des modèles pré-enregistrés [cite: 132]
        model = pickle.load(open('model_besoin1.pkl', 'rb'))
        scaler = pickle.load(open('scaler_besoin1.pkl', 'rb'))
        mapping = pickle.load(open('mapping_clusters.pkl', 'rb'))

        # Saisie des caractéristiques
        h_input = input("Entrez la hauteur de l'arbre (m) : ")
        d_input = input("Entrez le diamètre du tronc (cm) : ")
        
        h = float(h_input.replace(',', '.'))
        d = float(d_input.replace(',', '.'))

        print("\nCalcul de la catégorie...")
        time.sleep(0.5)

        # Transformation et Prédiction
        data_scaled = scaler.transform(np.array([[h, d]]))
        cluster_id = model.predict(data_scaled)[0]
        
        print(f"\n>>> RÉSULTAT : Cet arbre est classé : **{mapping[cluster_id]}**")

    except FileNotFoundError:
        print("Erreur : Modèles manquants. Exécutez train_besoin1.py.")
    except ValueError:
        print("Erreur : Veuillez entrer des nombres valides.")

if __name__ == "__main__":
    main()