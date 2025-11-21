# app.py

import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .approved {
        color: #28a745;
        font-weight: bold;
    }
    .rejected {
        color: #dc3545;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    try:
        return joblib.load("loan_pipeline.joblib")
    except FileNotFoundError:
        st.error("❌ Modèle non trouvé. Veuillez vérifier que 'loan_pipeline.joblib' existe.")
        return None


# Chargement du modèle
model = load_model()

# Sidebar pour la navigation
with st.sidebar:
    st.title("💰 Loan Predictor")
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Prédiction", "📊 Documentation", "ℹ️ À propos"])

    st.markdown("---")
    st.markdown("### Informations")
    st.info("""
    Cette application prédit la probabilité d'approbation de prêts basée sur des données historiques.
    """)

# Page principale de prédiction
if page == "🏠 Prédiction":
    # En-tête
    st.markdown('<h1 class="main-header">🏦 Loan Approval Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Entrez les détails du demandeur pour prédire l'approbation du prêt")

    # Layout en colonnes
    col1, col2 = st.columns([2, 1])

    with col1:
        with st.container():
            st.subheader("📝 Informations personnelles")
            col1a, col1b, col1c = st.columns(3)

            with col1a:
                gender = st.selectbox("Genre", ["Male", "Female"])
                education = st.selectbox("Éducation", ["Graduate", "Not Graduate"])

            with col1b:
                married = st.selectbox("Marié(e)", ["Yes", "No"])
                self_employed = st.selectbox("Indépendant", ["Yes", "No"])

            with col1c:
                dependents = st.selectbox("Personnes à charge", ["0", "1", "2", "3+"])
                property_area = st.selectbox("Zone de propriété", ["Urban", "Semiurban", "Rural"])

            st.subheader("💰 Informations financières")
            col2a, col2b, col2c = st.columns(3)

            with col2a:
                applicant_income = st.number_input(
                    "Revenu du demandeur ($)",
                    min_value=0,
                    step=500,
                    value=5000,
                    help="Revenu mensuel du demandeur principal"
                )

            with col2b:
                coapplicant_income = st.number_input(
                    "Revenu du co-demandeur ($)",
                    min_value=0,
                    step=500,
                    value=0,
                    help="Revenu mensuel du co-demandeur"
                )

            with col2c:
                loan_amount = st.number_input(
                    "Montant du prêt (k$)",
                    min_value=0,
                    step=10,
                    value=100,
                    help="Montant en milliers de dollars"
                )

            st.subheader("📄 Détails du prêt")
            col3a, col3b = st.columns(2)

            with col3a:
                loan_amount_term = st.selectbox(
                    "Durée du prêt (jours)",
                    [360, 180, 480, 300, 240, 120, 84],
                    help="Durée de remboursement en jours"
                )

            with col3b:
                credit_history = st.selectbox(
                    "Historique de crédit",
                    [1.0, 0.0],
                    format_func=lambda x: "✅ Bon historique" if x == 1.0 else "❌ Historique faible",
                    help="1.0 = bon historique de crédit, 0.0 = historique faible"
                )

    with col2:
        st.subheader("🎯 Prédiction")

        # Bouton de prédiction
        predict_btn = st.button(
            "🚀 Calculer la probabilité d'approbation",
            use_container_width=True,
            type="primary"
        )

        if predict_btn and model is not None:
            # Création du DataFrame d'entrée
            input_df = pd.DataFrame([{
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_amount_term,
                'Credit_History': credit_history,
                'Property_Area': property_area
            }])

            try:
                # Prédiction de probabilité
                prob = model.predict_proba(input_df)[0][1]
                decision = "Approved" if prob >= 0.5 else "Not Approved"

                # Affichage des résultats
                with st.container():
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

                    # Jauge de probabilité
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Probabilité d'approbation (%)"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightcoral"},
                                {'range': [30, 70], 'color': "lightyellow"},
                                {'range': [70, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                    # Décision
                    if decision == "Approved":
                        st.markdown(f'<h2 class="approved">✅ Prêt Approuvé</h2>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<h2 class="rejected">❌ Prêt Non Approuvé</h2>', unsafe_allow_html=True)

                    # Métriques
                    col_met1, col_met2 = st.columns(2)
                    with col_met1:
                        st.metric("Probabilité d'approbation", f"{prob:.1%}")
                    with col_met2:
                        st.metric("Seuil de décision", "50%")

                    st.markdown('</div>', unsafe_allow_html=True)

                    # Recommandations
                    if decision == "Not Approved" and prob < 0.5:
                        st.warning("""
                        **Recommandations pour améliorer l'approbation :**
                        - Améliorer l'historique de crédit
                        - Augmenter le revenu du demandeur/co-demandeur
                        - Réduire le montant du prêt demandé
                        """)

            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {str(e)}")

# Page de documentation
elif page == "📊 Documentation":
    st.title("📊 Documentation du modèle")

    tab1, tab2, tab3 = st.tabs(["📋 Variables", "📈 Importance des caractéristiques", "ℹ️ Guide"])

    with tab1:
        st.subheader("Description des variables")

        variables_data = {
            "Variable": [
                "Gender", "Married", "Dependents", "Education", "Self_Employed",
                "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                "Loan_Amount_Term", "Credit_History", "Property_Area"
            ],
            "Type": [
                "Catégorielle", "Catégorielle", "Catégorielle", "Catégorielle", "Catégorielle",
                "Numérique", "Numérique", "Numérique", "Numérique", "Binaire", "Catégorielle"
            ],
            "Description": [
                "Genre du demandeur",
                "Statut marital",
                "Nombre de personnes à charge",
                "Niveau d'éducation",
                "Statut d'indépendant",
                "Revenu mensuel du demandeur",
                "Revenu mensuel du co-demandeur",
                "Montant du prêt demandé (en milliers)",
                "Durée du prêt en jours",
                "Historique de crédit (1=bon, 0=faible)",
                "Zone géographique de la propriété"
            ],
            "Impact": [
                "Faible", "Moyen", "Moyen", "Moyen", "Moyen",
                "Élevé", "Élevé", "Élevé", "Moyen", "Très élevé", "Moyen"
            ]
        }

        df_variables = pd.DataFrame(variables_data)
        st.dataframe(df_variables, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Importance des caractéristiques")

        col_img, col_desc = st.columns([2, 1])

        with col_img:
            try:
                image = Image.open("feature_importance.png")
                st.image(image, caption="Importance des caractéristiques", use_container_width=True)
            except FileNotFoundError:
                st.warning("📊 Le graphique d'importance des caractéristiques n'est pas disponible.")

                # Graphique simulé à des fins de démonstration
                features = ['Credit_History', 'ApplicantIncome', 'LoanAmount', 'CoapplicantIncome',
                            'Property_Area', 'Loan_Amount_Term', 'Education', 'Married']
                importance = [0.35, 0.18, 0.15, 0.12, 0.08, 0.06, 0.04, 0.02]

                fig = px.bar(
                    x=importance,
                    y=features,
                    orientation='h',
                    title="Importance des caractéristiques (exemple)",
                    labels={'x': 'Importance', 'y': 'Caractéristiques'}
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_desc:
            st.info("""
            **Interprétation :**

            - **Credit_History** : Facteur le plus important
            - **Revenus** : Contribution significative à la décision
            - **Montant du prêt** : Impact négatif si trop élevé
            - **Autres facteurs** : Influence modérée à faible
            """)

    with tab3:
        st.subheader("Guide d'utilisation")

        st.markdown("""
        ### Comment utiliser cette application :

        1. **Remplir les informations** dans le formulaire de prédiction
        2. **Cliquer sur le bouton** de calcul de probabilité
        3. **Analyser les résultats** et les recommandations

        ### Seuil de décision :
        - ≥ 50% : Prêt approuvé
        - < 50% : Prêt refusé

        ### Facteurs clés d'approbation :
        - Historique de crédit impeccable
        - Revenus stables et suffisants
        - Montant du prêt proportionnel aux revenus
        """)

# Page À propos
elif page == "ℹ️ À propos":
    st.title("ℹ️ À propos de l'application")

    col_about, col_tech = st.columns(2)

    with col_about:
        st.subheader("Description")
        st.markdown("""
        Cette application de prédiction d'approbation de prêts utilise l'apprentissage automatique
        pour évaluer la probabilité qu'une demande de prêt soit approuvée basée sur des données historiques.

        **Fonctionnalités principales :**
        - Interface intuitive pour la saisie des données
        - Prédiction en temps réel avec visualisations
        - Explications détaillées des résultats
        - Documentation complète du modèle
        """)

    with col_tech:
        st.subheader("Technologies utilisées")
        st.markdown("""
        - **Streamlit** : Interface utilisateur
        - **Scikit-learn** : Modèle de machine learning
        - **Plotly** : Visualisations interactives
        - **Pandas** : Traitement des données
        - **Joblib** : Sauvegarde du modèle
        """)

    st.markdown("---")
    st.info("""
    ⚠️ **Disclaimer** : Cette prédiction est basée sur des données historiques et peut ne pas refléter 
    exactement les décisions réelles des institutions financières. Consultez toujours un conseiller financier 
    professionnel pour des décisions importantes.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Loan Approval Predictor © 2025 | Application de prédiction de prêts"
    "</div>",
    unsafe_allow_html=True
)