import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as express
import warnings
warnings.filterwarnings("ignore")

## configuration 
st.set_page_config(page_title="TrendApp", page_icon=":shark:", layout='wide')



# Fonction pour charger les données
@st.cache_data(persist=True)
def loading_data(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file)
        else:
            st.warning("Veuillez télécharger un fichier avec une extension .csv ou .xlsx.")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        return None

# Fonction pour visualiser les données
def data_visualisation(data):
    st.write('Aperçu des données :', data.sample(5))
    
    # Choix du type de visualisation
    type_visualisation = st.selectbox(
        "Choisissez votre type de visualisation", 
        ['Distribution Numérique', "Distribution des Variables Catégorielles", 
         "Relation entre les Variables", "Matrice de Corrélation"]
    )
    
    # Distribution des variables numériques
    if type_visualisation == "Distribution Numérique":
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
        if not numeric_features.empty:
            n_features = len(numeric_features)
            n_rows = (n_features + 1) // 2
            fig, axes = plt.subplots(n_rows, 2, figsize=(12, 10))
            fig.suptitle("Distribution des variables numériques", fontsize=16)
            
            for idx, variable in enumerate(numeric_features):
                row = idx // 2
                col = idx % 2
                ax = axes[row, col] if n_features > 1 else axes
                ax.hist(data[variable].dropna(), bins=30)
                ax.set_title(f"Distribution de {variable}")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            st.pyplot(fig)
        else:
            st.warning("Aucune variable numérique disponible pour cette visualisation.")

    # Distribution des variables catégorielles
    elif type_visualisation == "Distribution des Variables Catégorielles":
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        if not categorical_columns.empty:
            variable = st.selectbox("Sélectionnez une variable catégorielle", categorical_columns)
            if variable:
                fig, ax = plt.subplots()
                sns.countplot(data=data, x=variable, ax=ax)
                ax.set_title(f"Distribution de {variable}")
                st.pyplot(fig)
        else:
            st.warning("Aucune variable catégorielle disponible pour cette visualisation.")

    # Nuage de points pour les relations entre deux variables
    elif type_visualisation == "Relation entre les Variables":
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_columns) > 1:
            x_var = st.selectbox("Sélectionnez la variable pour l'axe X", numeric_columns)
            y_var = st.selectbox("Sélectionnez la variable pour l'axe Y", numeric_columns)
            hue_var = st.selectbox("Choisissez une variable catégorielle pour la couleur (optionnel)", 
                                   [None] + list(data.select_dtypes(include=['object', 'category']).columns))
            
            if x_var and y_var:
                fig, ax = plt.subplots()
                sns.scatterplot(data=data, x=x_var, y=y_var, hue=hue_var, ax=ax)
                ax.set_title(f"Relation entre {x_var} et {y_var}")
                st.pyplot(fig)
        else:
            st.warning("Pas assez de variables numériques pour afficher une relation.")

    # Matrice de corrélation
    elif type_visualisation == "Matrice de Corrélation":
        numeric_data = data.select_dtypes(include=['float64', 'int64']).dropna()
        if not numeric_data.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = numeric_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Matrice de Corrélation")
            st.pyplot(fig)
        else:
            st.warning("Pas de données numériques disponibles pour une matrice de corrélation.")

# Fonction principale pour gérer la visualisation
def visualize():
    uploaded_file = st.sidebar.file_uploader("Choisissez un fichier", type=["csv", "xlsx"], key='uploader_visualize')
    if uploaded_file:
        data = loading_data(uploaded_file)
        if data is not None and not data.empty:
            data_visualisation(data)
        else:
            st.warning("Les données sont vides ou mal formatées.")

# Application Streamlit
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Aller à :", ["Accueil", "Visualiser"])

if menu == "Accueil":
    st.title('Visualisation des tendances du dioxyde de carbone (CO2) 1990-2020')
    st.write("""
    **TrendApp** est une application interactive qui vous permet de :
    - Visualiser les données de CO2 émises entre 1990 et 2020.
    - Explorer différentes visualisations interactives.
    """)
    st.image("pages/trendapp.png", caption="TrendApp", use_container_width=True)

elif menu == "Visualiser":
    st.title("Visualisez vos données facilement")
    visualize()


st.markdown("Réalisé par :")
st.write('Eric Dona KOULODJI, Physicien Théoricien, Data Scientist')