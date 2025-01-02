##  Projet 1 : Emissions du Dioxyde de carbone de 1990-2020

![image](https://github.com/user-attachments/assets/d85a7599-9d03-4add-a951-f8b2f07b7096)

# A- ** First Step **
Ce projet consiste à analyser les données sur l'émissions de C02 récueillies par les appareils satelitaires et les capteurs de surveillance pour prédire les émissions en Co2 en Afrique et sur le reste du continent.
Les données sont disponibles sur kaggle à l'adresses: https://www.kaggle.com/datasets/ravindrasinghrana/carbon-co2-emissions .
## 1- Importation des données
## 2- Analyse univariée des données
## 3- Analyse Bivariée et multivariée 
## 4- Réaliser la PCA (Principales Components Analysis )

### Technologies : 
    ** Pandas
    ** Numpy
    ** plotly
    ** Matplotlib
    ** Seaborn 
    ** Scikit-learn

# B- ** Second Step **

Mettre en place une application streamlit interactive et dynamique qui permet de visulaiser les données et explorer les différentes visualisations possibles.
Cette application est : # TrendApp : une application de visualisations des tendances et qui vous permettra de prendre des décisions claires.
### Comment utilisé TrendApp
 * Pour ceux qui souhaite reutiliser le code source , il vous suffit de vous rendre sur : https://github.com/dona-eric/Emissions-du-Dioxyde-de-carbone-CO2 et cloner le repo en cherchant le 
fichier .py (TrendApp.py )
* Le deploiement se fera d'ici peu et peut etre accessible et pret à l'usage.
* Lancer le serveur : streamlit run TrendApp.py
# * Technologies:
    ** streamlit
    ** plotly


## Projet 2: Energy Data collect 1990-2020

### A- First Step

### B- Second Step 

  * Mettre en place une API qui faire la requete des données et renvoie via une interface streamlit le résultat de la prediction.
  
  * Je vous explique ***
  Après analyse des données et la modélisations, un pipeline de transformation des données a été construit afin d'appliquer les traitements possibles sur les données ; Un modèle robuste bien plus meilleur que les modèles classiques de LinearRegression ou Ridge de régularisation. 
  Avec une validation croisée et comparaison approfondit le modèle performant pour resister meme aux outliers (valeurs aberrantes) est un modèle de choix : HubRegressor.

  * Avec la création d'une api à l'aide FastAPI, on fait passer les données via une interface construite à l'aide de streamlit. Ehh pfff la magie est opérée.

### Technologies utilisées :
    * FastAPI
    * Streamlit
    * SCikit-learn
    * Pandas
    * Numpy, Plotly, Seaborn
    * DataViz
    * Dtale

### How can use it ?
You can clone the repository with the link [Repo cloning: ](https://github.com/dona-eric/Emissions-du-Dioxyde-de-carbone-CO2)

Launch the server FastAPI with : uvicorn main:app --reload
Visit the url in local host : http://127.0.0.1:8000/predict

Launch the server of Streamlit with : streamlit run data_app.py
visit the url in Local URL: http://localhost:8501

*** You can run the scripts python to get the results. The ideas inspired to the course of microsoft learn where i get the certificate professionnel for entraining and get the solutions to deploy the model machine learning on Azure Machine Learning. Thanks Microsoft !

*** For anyone collaborations, you can write me or send-me the email if you visite my profil .
## Author :
Eric D. KOULODJI , Data Scientist Junior, Physics Theorics, IA Enthusiast.
