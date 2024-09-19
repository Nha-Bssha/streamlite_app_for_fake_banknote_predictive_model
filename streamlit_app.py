# app_ml.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Ajouter le logo
st.markdown(
    """
    <style>
    .banner-img {
        width: 100%;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True
)

# Afficher l'image comme bannière
st.image("logo.png", use_column_width=True)

# ----------------------------
# Titre de l'application
st.title("Modèle de détection automatique de faux billets de banque")

# Chargement des données
st.header("Chargement des données")

# Téléchargement des fichiers CSV (sans limite de taille)
train_file = st.file_uploader(
    "Télécharger le fichier d'entraînement (CSV)", type="csv")
test_file = st.file_uploader(
    "Télécharger le fichier de test (CSV)", type="csv")

if train_file is not None and test_file is not None:
    data_prod = pd.read_csv(train_file)
    data_test = pd.read_csv(test_file)

    st.write("Aperçu des données d'entraînement")
    st.write(data_prod.head())

    st.write("Aperçu des données de test")
    st.write(data_test.head())

    # Vérification des valeurs manquantes et doublons
    st.subheader("Vérification des données")
    if data_test.isna().sum().sum() != 0:
        st.warning("Le fichier de test contient des valeurs manquantes.")
    else:
        st.success("Pas de valeurs manquantes dans le fichier de test.")

    if data_test.duplicated().sum() != 0:
        st.warning("Le fichier de test contient des doublons.")
    else:
        st.success("Pas de doublons dans le fichier de test.")

    # Sélection des variables pour la régression logistique
    st.subheader("Prétraitement des données")
    x_train_logreg = data_prod[['height_right',
                                'margin_low', 'margin_up', 'length']].values
    y_train = data_prod['is_genuine']
    x_test_logreg = data_test[['height_right',
                               'margin_low', 'margin_up', 'length']].values

    # Standardisation des données
    st.write("Standardisation des données")
    std_scaler_logreg = StandardScaler()
    x_train_logreg_scaled = std_scaler_logreg.fit_transform(x_train_logreg)
    x_test_logreg_scaled = std_scaler_logreg.transform(x_test_logreg)

    # Entraînement du modèle
    st.subheader("Entraînement du modèle de régression logistique")
    best_params_saga = {'C': 0.001, 'class_weight': None,
                        'max_iter': 500, 'solver': 'saga', 'random_state': 42}
    estimator_final = LogisticRegression(**best_params_saga)
    estimator_final.fit(x_train_logreg_scaled, y_train.values.ravel())

    st.success("Modèle entraîné avec succès!")

    # Prédiction sur les données de test
    st.subheader("Prédictions sur les données de test")
    results = data_test.copy()
    results['proba'] = estimator_final.predict_proba(x_test_logreg_scaled)[
        :, 1].ravel()
    results['labels_pred_reglog'] = results['proba'] > 0.5

    # Affichage des résultats
    st.write("Résultats des prédictions")
    for i, j in zip(results["labels_pred_reglog"], results.index):
        if i:
            st.write(f"Le billet ref. {j} est un vrai billet.")
        else:
            st.write(f"Le billet ref. {j} est un faux billet.")


else:
    st.info(
        "Veuillez télécharger les fichiers CSV d'entraînement et de test pour démarrer.")
