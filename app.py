import streamlit as st
import requests
import pandas as pd
import numpy as np
import json

# Configuration de la page
st.set_page_config(
    page_title="Outil de Scoring Crédit",
    page_icon="💰",
    layout="wide"
)

# URL de l'API (remplacez par votre URL ngrok)
API_URL = "https://credit.ngrok-free.app"

# Liste des features du modèle
FEATURES = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'DAYS_BIRTH', 'CODE_GENDER_M', 'CNT_CHILDREN',
    'DAYS_EMPLOYED', 'DAYS_EMPLOYED_PERCENT', 'NAME_INCOME_TYPE_Working',
    'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT',
    'REG_CITY_NOT_WORK_CITY', 'FLAG_OWN_REALTY', 'OCCUPATION_TYPE_Laborers'
]

# Descriptions des features pour une meilleure expérience utilisateur
FEATURE_DESCRIPTIONS = {
    'AMT_INCOME_TOTAL': 'Revenu annuel total',
    'AMT_CREDIT': 'Montant du crédit demandé',
    'AMT_ANNUITY': 'Montant des paiements annuels',
    'CREDIT_INCOME_PERCENT': 'Ratio crédit/revenu',
    'ANNUITY_INCOME_PERCENT': 'Ratio annuité/revenu',
    'CREDIT_TERM': 'Durée du crédit (années)',
    'EXT_SOURCE_1': 'Score externe 1 (0-1)',
    'EXT_SOURCE_2': 'Score externe 2 (0-1)',
    'EXT_SOURCE_3': 'Score externe 3 (0-1)',
    'DAYS_BIRTH': 'Age en jours (négatif)',
    'CODE_GENDER_M': 'Genre (1=Homme, 0=Femme)',
    'CNT_CHILDREN': 'Nombre d\'enfants',
    'DAYS_EMPLOYED': 'Jours d\'emploi (négatif)',
    'DAYS_EMPLOYED_PERCENT': 'Ratio emploi/âge',
    'NAME_INCOME_TYPE_Working': 'Travailleur salarié (1=Oui, 0=Non)',
    'REGION_RATING_CLIENT_W_CITY': 'Note région+ville (1-3)',
    'REGION_RATING_CLIENT': 'Note région (1-3)',
    'REG_CITY_NOT_WORK_CITY': 'Ville travail ≠ résidence (1=Oui, 0=Non)',
    'FLAG_OWN_REALTY': 'Propriétaire (1=Oui, 0=Non)',
    'OCCUPATION_TYPE_Laborers': 'Ouvrier (1=Oui, 0=Non)'
}

# Fonction pour appeler l'API
def predict_credit_risk(features):
    try:
        response = requests.post(
            f"{API_URL}/predict", 
            json={"features": features},
            auth=("admin", "credit_scoring_tool")  # Ajout des identifiants ici
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur de connexion: {str(e)}")
        return None

# Interface utilisateur
st.title("📊 Outil d'Évaluation de Risque de Crédit")
st.markdown("Cet outil utilise un modèle XGBoost pour prédire le risque de défaut de paiement.")

# Création de colonnes pour organiser les inputs
col1, col2 = st.columns(2)

# Initialisation des valeurs par défaut (exemple de client)
if 'feature_values' not in st.session_state:
    st.session_state.feature_values = {
        'AMT_INCOME_TOTAL': 150000.0,
        'AMT_CREDIT': 600000.0,
        'AMT_ANNUITY': 30000.0,
        'CREDIT_INCOME_PERCENT': 4.0,
        'ANNUITY_INCOME_PERCENT': 0.2,
        'CREDIT_TERM': 20.0,
        'EXT_SOURCE_1': 0.7,
        'EXT_SOURCE_2': 0.65,
        'EXT_SOURCE_3': 0.8,
        'DAYS_BIRTH': -14000.0,  # environ 38 ans
        'CODE_GENDER_M': 1.0,
        'CNT_CHILDREN': 2.0,
        'DAYS_EMPLOYED': -3650.0,  # environ 10 ans
        'DAYS_EMPLOYED_PERCENT': 0.26,
        'NAME_INCOME_TYPE_Working': 1.0,
        'REGION_RATING_CLIENT_W_CITY': 2.0,
        'REGION_RATING_CLIENT': 2.0,
        'REG_CITY_NOT_WORK_CITY': 0.0,
        'FLAG_OWN_REALTY': 1.0,
        'OCCUPATION_TYPE_Laborers': 0.0
    }

# Organisation des features en catégories pour une meilleure UX
with col1:
    st.subheader("Informations financières")
    for feature in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
                   'CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM']:
        st.session_state.feature_values[feature] = st.number_input(
            f"{FEATURE_DESCRIPTIONS[feature]}", 
            value=st.session_state.feature_values[feature],
            step=0.01
        )
    
    st.subheader("Scores externes")
    for feature in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        st.session_state.feature_values[feature] = st.slider(
            f"{FEATURE_DESCRIPTIONS[feature]}", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.feature_values[feature],
            step=0.01
        )

with col2:
    st.subheader("Informations personnelles")
    age_years = abs(int(st.session_state.feature_values['DAYS_BIRTH'])) // 365
    age = st.slider("Âge (années)", 18, 100, age_years)
    st.session_state.feature_values['DAYS_BIRTH'] = -age * 365
    
    for feature in ['CODE_GENDER_M', 'CNT_CHILDREN', 'FLAG_OWN_REALTY']:
        if feature == 'CODE_GENDER_M':
            st.session_state.feature_values[feature] = float(st.selectbox(
                "Genre", 
                options=["Femme", "Homme"], 
                index=int(st.session_state.feature_values[feature])
            ) == "Homme")
        elif feature == 'FLAG_OWN_REALTY':
            st.session_state.feature_values[feature] = float(st.checkbox(
                "Propriétaire d'un bien immobilier", 
                value=bool(st.session_state.feature_values[feature])
            ))
        else:
            st.session_state.feature_values[feature] = st.number_input(
                f"{FEATURE_DESCRIPTIONS[feature]}", 
                min_value=0.0, 
                max_value=10.0, 
                value=st.session_state.feature_values[feature],
                step=1.0
            )
    
    st.subheader("Informations professionnelles")
    employed_years = abs(int(st.session_state.feature_values['DAYS_EMPLOYED'])) // 365
    employed = st.slider("Années d'emploi", 0, 50, employed_years)
    st.session_state.feature_values['DAYS_EMPLOYED'] = -employed * 365
    st.session_state.feature_values['DAYS_EMPLOYED_PERCENT'] = employed / age if age > 0 else 0
    
    for feature in ['NAME_INCOME_TYPE_Working', 'OCCUPATION_TYPE_Laborers']:
        if feature == 'NAME_INCOME_TYPE_Working':
            st.session_state.feature_values[feature] = float(st.checkbox(
                "Travailleur salarié", 
                value=bool(st.session_state.feature_values[feature])
            ))
        else:
            st.session_state.feature_values[feature] = float(st.checkbox(
                "Ouvrier", 
                value=bool(st.session_state.feature_values[feature])
            ))
    
    st.subheader("Informations régionales")
    for feature in ['REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT', 'REG_CITY_NOT_WORK_CITY']:
        if feature in ['REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT']:
            st.session_state.feature_values[feature] = st.slider(
                f"{FEATURE_DESCRIPTIONS[feature]}", 
                min_value=1.0, 
                max_value=3.0, 
                value=st.session_state.feature_values[feature],
                step=1.0
            )
        else:
            st.session_state.feature_values[feature] = float(st.checkbox(
                "Travaille dans une ville différente de celle de résidence", 
                value=bool(st.session_state.feature_values[feature])
            ))

# Bouton de soumission
st.markdown("---")
submit_col1, submit_col2, submit_col3 = st.columns([1, 1, 1])

with submit_col2:
    if st.button("Évaluer le risque de crédit", type="primary", use_container_width=True):
        # Conversion des valeurs en liste en respectant l'ordre des features
        features_list = [st.session_state.feature_values[feature] for feature in FEATURES]
        
        # Affichage d'un spinner pendant l'appel API
        with st.spinner("Calcul du score de risque en cours..."):
            result = predict_credit_risk(features_list)
            
        if result:
            # Affichage des résultats
            st.markdown("---")
            risk_col1, risk_col2 = st.columns(2)
            
            with risk_col1:
                st.metric(
                    label="Probabilité de défaut", 
                    value=f"{result['probability']*100:.2f}%"
                )
                
                # Affichage visuel du score
                st.progress(result['probability'])
                
            with risk_col2:
                decision = result.get('decision', 'Inconnu')
                if decision == "HIGH RISK":
                    st.error("⚠️ RISQUE ÉLEVÉ")
                    st.markdown("Le modèle indique un risque significatif de défaut de paiement.")
                else:
                    st.success("✅ RISQUE FAIBLE")
                    st.markdown("Le modèle indique un risque limité de défaut de paiement.")
            
            # Ajout d'explications sur les facteurs influençant la décision
            st.markdown("---")
            st.subheader("Facteurs influençant la décision")
            st.markdown("""
            Selon notre modèle, les facteurs suivants ont le plus d'impact sur l'évaluation du risque:
            
            1. **Sources externes (EXT_SOURCE)** - Les scores externes sont généralement les prédicteurs les plus puissants
            2. **Ratio crédit/revenu** - Un ratio élevé indique une charge financière importante
            3. **Stabilité professionnelle** - Une longue période d'emploi réduit généralement le risque
            """)

# Pied de page
st.markdown("---")
st.caption("Modèle XGBoost entraîné sur 20 features | Développé avec FastAPI et Streamlit")
