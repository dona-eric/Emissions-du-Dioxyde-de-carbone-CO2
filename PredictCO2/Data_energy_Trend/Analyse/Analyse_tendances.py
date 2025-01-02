#!/usr/bin/env python
# coding: utf-8

# Objectif : Analyser les tendances mondiales en matières d'énergie ; une analyse des données receuillies entre 1990-2020. Prévoir les tendances mondiales en matière d'énergie et d'émissions de CO2.  

# imporation des librarys

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings(action='ignore')


# In[3]:


data = pd.read_csv('/home/dona-erick/Projet CO2/PredictCO2/data/Energy data 1990 - 2020.csv')
data.head()


# In[4]:


data.info()


# In[5]:


print(data.columns), print(data.dtypes), print(data.isnull().sum())


# In[6]:


data.describe().T.mean()


# In[7]:


data.shape


# In[8]:


data.duplicated().sum()


# ## Analyse Univariée

# In[9]:


data.sample(5
            )


# In[10]:


data['Region'].value_counts().plot(kind = 'hist',bins = 20)


# In[11]:


plt.figure(figsize=(12, 10))

for var in data.select_dtypes(exclude = ['int64', "float64"]).columns:
    print(var)
    sns.countplot(data = data, y=var)
    plt.title(f'Distribution par {var}')
    plt.tight_layout()
    plt.savefig('images')
    plt.show()


# In[12]:


numeric_col = data.select_dtypes(include='number')
cat_cols = data.select_dtypes(exclude =['float64', 'int64'])

for col in numeric_col.columns:
    print(col)
    sns.boxplot(data =data, y=col, palette = 'Set2')
    plt.title(f'Distribution de {col}')
    plt.tight_layout()
    plt.show()


# In[13]:


plt.figure(figsize=(20, 10))
sns.heatmap(data=numeric_col.corr(), annot=True, cmap ='coolwarm', fmt='.2f')


# In[14]:


plt.figure(figsize=(12, 10))
sns.pairplot(data=numeric_col)


# ## Analyse Bivariée et Mutlivariée

# In[15]:


data.head(2)


# In[16]:


fig, ax = plt.subplots(figsize=(12, 10))

sns.scatterplot(data = data, x ="CO2 intensity at constant purchasing power parities (kCO2/$15p)", 
                y ="Average CO2 emission factor (tCO2/toe)", hue='Region', ax=ax )
ax.set_xlabel('Intensité de CO2 Power')
ax.set_ylabel('CO2 moyenne emises')
ax.set_title('Distribution des variables')
plt.show()


# In[17]:


fig, ax = plt.subplots(figsize=(12, 10))

sns.scatterplot(data = data, x ="CO2 emissions from fuel combustion (MtCO2)", 
                y ="Average CO2 emission factor (tCO2/toe)", hue='Region', ax=ax )
ax.set_xlabel('CO2 emissions de combustions')
ax.set_ylabel('CO2 moyenne emises')
ax.set_title('Distribution des variables')
plt.show()


# ## A propos de l'ensembles des données 

# 
# ###  Annuaire mondial de l'énergie 1990 - 2020.
# 
# * Abréviations:
# 
#         Mtep millions de tonnes d'équivalent pétrole (103 tep)
#         tep tonnes d'équivalent pétrole
#         koe kilo d'équivalent pétrole (10-3 tep)
#         Mt millions de tonnes
#         bcm milliards de mètres cubes (109 mètres cubes)
#         TWh térawattheure
#         tCO2 tonnes de dioxyde de carbone
#         kCO2 kilogramme de dioxyde de carbone (10-3 tCO2)
# 
# * Glossaire: 
# 
#         Balance commerciale : La balance commerciale est la différence entre les importations et les exportations. Le solde d'un exportateur net apparaît comme une valeur négative (-). Le solde des zones géographiques et géopolitiques est simplement la somme des soldes commerciaux de tous les pays.
# 
#         Emissions de CO2 issues de la combustion d'énergies fossiles : Les émissions de CO2 ne couvrent que les émissions issues de la combustion d'énergies fossiles (charbon, pétrole et gaz). Elles sont calculées selon la méthodologie de la CCNUCC. On présente ici l'approche de référence, c'est-à-dire la somme des émissions de CO2 de chaque énergie.
# 
#         Intensité de CO2 : L'intensité de CO2 est le rapport entre les émissions de CO2 provenant de la combustion de combustibles et le produit intérieur brut (PIB) mesuré en dollars américains constants à parité de pouvoir d'achat. Elle mesure le CO2 émis pour générer une unité de PIB. Le PIB est exprimé à taux de change et parité de pouvoir d'achat constants pour éliminer l'impact de l'inflation et refléter les différences de niveaux de prix généraux et relier la consommation d'énergie au niveau réel de l'activité économique. L'utilisation de taux de parité de pouvoir d'achat pour le PIB au lieu de taux de change augmente la valeur du PIB dans les régions où le coût de la vie est faible et diminue donc leur intensité énergétique.
# 
#         Pétrole brut : Le pétrole brut comprend tous les hydrocarbures liquides à raffiner : pétrole brut, liquides provenant du gaz naturel (LGN) et produits semi-raffinés.
# 
#         Production de pétrole brut, de charbon et de lignite : correspond à la production brute.
# 
#         Production d'électricité : La production d'électricité correspond à la production brute. Elle comprend la production publique (production des sociétés d'électricité privées et publiques) et les producteurs industriels pour leurs propres besoins, par tout type de centrale électrique (y compris la cogénération).
# 
#         Intensité énergétique du PIB à parités de pouvoir d'achat constantes : L'intensité énergétique est le rapport entre la consommation d'énergie primaire et le produit intérieur brut (PIB) mesuré en dollars américains constants à parités de pouvoir d'achat. Elle mesure la quantité totale d'énergie nécessaire pour générer une unité de PIB. Le PIB est exprimé à taux de change et parité de pouvoir d'achat constants pour éliminer l'impact de l'inflation et refléter les différences de niveaux de prix généraux et relier la consommation d'énergie au niveau réel de l'activité économique. L'utilisation des taux de parité de pouvoir d'achat pour le PIB au lieu des taux de change augmente la valeur du PIB dans les régions où le coût de la vie est faible et diminue donc leur intensité énergétique.
# 
#         Production de gaz naturel : La production de gaz naturel correspond à la production commercialisée (c'est-à-dire hors quantités torchées ou réinjectées).
# 
# * NGL : Liquides de gaz naturel
# 
#         Produits pétroliers : Les produits pétroliers sont tous les hydrocarbures liquides, obtenus par le raffinage du pétrole brut et du NGL et par le traitement du gaz naturel, en particulier la production de GPL (gaz de pétrole liquéfié) comprend le GPL issu des usines de séparation du gaz naturel. L'éthanol utilisé comme carburant au Brésil ainsi que les carburants dérivés du charbon en Afrique du Sud ne sont pas inclus dans les produits pétroliers.
# 
#         Part des énergies renouvelables dans la production d'électricité : Rapport entre la production d'électricité issue des énergies renouvelables (hydraulique, éolien, géothermie et solaire) et la production totale d'électricité.
# 
#         Part de l'éolien et du solaire dans la production d'électricité : Électricité produite à partir de l'énergie éolienne et solaire divisée par la production totale d'électricité.
# 
#         Part de l’électricité dans la consommation finale totale d’énergie : Demande finale d’électricité divisée par la consommation finale totale d’énergie.
# 
#         Production primaire totale : La production primaire évalue la quantité de ressources énergétiques naturelles (« sources d'énergie primaires ») extraites ou produites. Pour le gaz naturel, les quantités brûlées ou réinjectées sont exclues. La production d'électricité hydraulique, géothermique, nucléaire et éolienne est considérée comme production primaire.
# 
#         Consommation totale d'énergie : La consommation totale d'énergie est le solde de la production primaire, du commerce extérieur, des soutes maritimes et des variations de stocks. La consommation totale d'énergie inclut la biomasse. Pour le monde, les soutes maritimes sont incluses. Cela induit un écart avec la somme des régions.
# 
#         Facteur d'émission moyen de CO2 : Le facteur d'émission moyen de CO2 (facteur carbone) est calculé en faisant le rapport entre les émissions et la consommation d'énergie primaire.

# In[18]:


fig, ax = plt.subplots()
sns.lineplot(data = data, x = 'Year', y = 'CO2 emissions from fuel combustion (MtCO2)',
             hue='Region', ax = ax)
ax.set_title('CO2 emisses par la combustion fuel par année dans chaque region')
ax.set_xlabel('Year')
ax.set_ylabel('CO2 emissions from fuel combustion (MtCO2)')
plt.show()


# In[19]:


### totla enrgy produite dans la region africaine
total_energy_produite_africa = data.groupby('Region')['Total energy production (Mtoe)'].mean()
total_energy_produite_africa.plot(kind = 'bar')


# In[20]:


fig, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(data = data, x = 'Year', y = 'CO2 emissions from fuel combustion (MtCO2)',
             hue='Region', ax = ax)
ax.set_title('CO2 emisses par la combustion fuel par année dans chaque region')
ax.set_xlabel('Year')
ax.set_ylabel('CO2 emissions from fuel combustion (MtCO2)')
plt.show()


fig, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(data = data, x = 'Year', y = 'Average CO2 emission factor (tCO2/toe)',
             hue='Region', ax = ax)
ax.set_title('CO2 emisses par la combustion fuel par année dans chaque region')
ax.set_xlabel('Year')
ax.set_ylabel('Average CO2 emission factor (tCO2/toe)	')
plt.show()



fig, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(data = data, x = 'Year', y = 'CO2 intensity at constant purchasing power parities (kCO2/$15p)',
             hue='Region', ax = ax)
ax.set_title('CO2 intensity at constant purchasing power parities (kCO2/$15p)')
ax.set_xlabel('Year')
ax.set_ylabel('CO2 emissions from fuel combustion (MtCO2)')
plt.show()


# In[21]:


def bar_label(axes, _type = "edge", rotation=0, axis=True):
    for container in axes.containers:
        axes.bar_label(container, label_type=_type, rotation=rotation)
    if axis:
        axes.set_xticklabels(axes.get_xticklabels(), rotation=90)


# In[22]:


region = data["Region"].value_counts()
dt = pd.DataFrame(region)
fig, axes = plt.subplots()
sns.barplot(x=dt.index[:20], y=dt.iloc[:20, 0], ax=axes)
bar_label(axes, "center", 90, True)
plt.show()


# In[23]:


group = data.groupby("Year")
val = group["CO2 emissions from fuel combustion (MtCO2)"].sum()
fig, axes = plt.subplots(figsize=(20, 6))
sns.barplot(x=val.index, y=val, ax=axes)
plt.title("Somme CO2 emissions from fuel combustion (MtCO2) par année")
bar_label(axes, "center", 90, True)
plt.show()


# In[24]:


group = data.groupby("Region")
val = group["CO2 emissions from fuel combustion (MtCO2)"].sum()
val = pd.DataFrame(val)
val = val.sort_values("CO2 emissions from fuel combustion (MtCO2)", ascending=False)
fig, axes = plt.subplots(figsize=(12, 6))
sns.barplot(x=val.index[:10], y=val.iloc[:10, 0], ax=axes)
plt.title('Total CO2 emissions from fuel combustion (MtCO2) by year')
bar_label(axes, "edge", 0, True)
plt.show()


# In[25]:


group = data.groupby("country")
val = group["CO2 emissions from fuel combustion (MtCO2)"].sum()
val = pd.DataFrame(val)
val = val.sort_values("CO2 emissions from fuel combustion (MtCO2)", ascending=False)
fig, axes = plt.subplots(figsize=(15, 10))
sns.barplot(x=val.index, y=val.iloc[:, 0], ax=axes)
plt.title('Total CO2 emissions from fuel combustion (MtCO2) by Country')
bar_label(axes, "edge", 0, True)
plt.show()


# In[26]:


cross = pd.crosstab(data["country"], data["Region"])
sns.heatmap(cross, annot=True)
plt.show()


# ## Preprocessing data

# In[27]:


data.isnull().sum()


# In[28]:


data.info()


# In[29]:


columns_transformed = ["Coal and lignite production (Mt)", "Share of wind and solar in electricity production (%)",
                       "Natural gas production (bcm)", "Coal and lignite domestic consumption (Mt)"]

for col in columns_transformed:
    data[col] = pd.to_numeric(data[col], errors= 'coerce')


# In[30]:


data.info()


# In[31]:


import missingno as msno
msno.matrix(data)


# In[32]:


#data.isna().sum()


# In[33]:


numeric_vars = data.select_dtypes(include = ['int64', "float64"])

correlation_data = numeric_vars.corr()
sns.heatmap(correlation_data, annot = True, cmap = 'coolwarm', fmt='.2f')


# In[34]:


## methode de pearson
plt.figure(figsize =(12, 10))
correlation_data = numeric_vars.corr(method='pearson')
sns.heatmap(correlation_data, annot = True, cmap = 'coolwarm', fmt='.2f')


# In[35]:


plt.figure(figsize=(20, 10))
correlation_data = numeric_vars.corr(method='spearman')
sns.heatmap(correlation_data, annot = True, cmap = 'coolwarm', fmt='.2f')
plt.title('Correlation de spearman')


# In[36]:


### création des histogrammees et test de normalité

for col in numeric_vars.columns:
    print(col)
    plt.figure(figsize=(10, 8))
    sns.histplot(data = numeric_vars, x=col, kde = True)
    plt.title(f'Histrogramme de {col}')
    plt.show()


# In[37]:


from scipy.stats import shapiro

# pour mes tests de normalité

test_normal = []
for var in numeric_vars:
    stat, p = shapiro(numeric_vars[var])
    normality = "Normal" if p > 0.05 else 'Not Normal'
    test_normal.append({"Variable": var,
                                      'p-value':p, 'Normality': normality})
    
    results = pd.DataFrame(test_normal)
print(results)


# ## Realiser la pca sur les données

# In[38]:


from sklearn.decomposition import PCA, KernelPCA
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator

X = data.select_dtypes(exclude ='object')
X


# In[39]:


df = data.copy()
imputer = SimpleImputer(strategy = 'median')
for col in df.select_dtypes(exclude = 'object').columns:
    df[col] = imputer.fit_transform(df[[col]])
    
df[col].isnull().sum()


# In[40]:


X = df.select_dtypes(exclude ='object')


# In[41]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder
pca = PCA()
X_scaled = StandardScaler().fit_transform(X)
X_pca = pca.fit(X_scaled)

variance_expliqué_ratio = np.round(pca.explained_variance_ratio_ *100)
valeur_propre = pca.explained_variance_
cumulated_sum = np.round(np.cumsum(variance_expliqué_ratio), 2)

data_pca = pd.DataFrame({
    'valeur_propre': valeur_propre,
    '% variance.expliqué': variance_expliqué_ratio,
    '%cumulated_variance': cumulated_sum
}, columns = ['valeur_propre', "% variance.expliqué", "%cumulated_variance"])

data_pca


# In[42]:


plt.figure(figsize=(10, 8))
plt.plot(range(1, len(cumulated_sum)+1), cumulated_sum, marker = 'o', linestyle = "--")
plt.title('variance expliquée en fonction du nombre de composantes')
plt.xlabel('Nombre de composantes principales')
plt.ylabel('Variance cumulée')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% de variance expliquée')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% de variance expliquée')
plt.legend(loc='best')
plt.grid()
plt.show()


# Pour appliquer la reduction de dimensionnalité sur les données sans perdre plus d'information, on puet choisir n=5 qui explique absoluement 70% de la variance.
# A partir de n =10 , on peut constater que la courbe cumulative commence par se stabilisr et capture plus de 90 % de la variance. 

# In[43]:


pca = PCA(n_components=4)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize = (12, 10))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# ## Traitement des données 

# ### valeurs manquantes 
# ### mettre en place un pipeline pour transformer ses données 
# ### make transformations columns avec columntransformers

# In[44]:


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer, MissingIndicator
from sklearn.compose import ColumnTransformer,make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split


import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression, SGDRegressor, ElasticNet, HuberRegressor, QuantileRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Transformation des données
def transformation_data(data):
    """Préparer les données en effectuant les transformations nécessaires."""
    
    X = data.drop('CO2 emissions from fuel combustion (MtCO2)', axis=1)
    y = data['CO2 emissions from fuel combustion (MtCO2)']

    # Identifier les variables numériques et catégorielles
    num_var = X.select_dtypes(exclude='object').columns
    cat_var = X.select_dtypes(include='object').columns

    # Pipeline pour les variables numériques
    pipeline_numeric = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('knn_imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    # Pipeline pour les variables catégorielles
    pipeline_categorical = Pipeline(steps=[
        ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Transformation des colonnes avec ColumnTransformer
    transformer = ColumnTransformer(transformers=[
        ('num', pipeline_numeric, num_var),
        ('cat', pipeline_categorical, cat_var)
    ])

    # Diviser les données en train et test
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=80)

    # Appliquer les transformations
    Xtrain_new, Xtest_new = transformer.fit_transform(Xtrain), transformer.transform(Xtest)
    
    return Xtrain_new, Xtest_new, ytrain, ytest, transformer

# 2. Entraînement et évaluation des modèles
def train_and_evaluate_model(estimator, Xtrain, Xtest, ytrain, ytest):
    """Entraîner un modèle et évaluer ses performances sur les ensembles d'entraînement et de test."""
    
    # Entraîner le modèle
    estimator.fit(Xtrain, ytrain)
    
    # Prédictions sur l'ensemble d'entraînement
    predict_train = estimator.predict(Xtrain)
    mae_train = mean_absolute_error(ytrain, predict_train)
    rmse_train = mean_squared_error(ytrain, predict_train, squared=False)  
    r2_train_score = r2_score(ytrain, predict_train)
    
    # Prédictions sur l'ensemble de test
    predict_test = estimator.predict(Xtest)
    mae_test = mean_absolute_error(ytest, predict_test)
    rmse_test = mean_squared_error(ytest, predict_test, squared=False)  
    r2_test_score = r2_score(ytest, predict_test)
    
    # Résultats sous forme de dictionnaire
    result_model_train = {
        "MAE": mae_train,
        "RMSE": rmse_train,
        "R2_score": r2_train_score
    }
    
    result_model_test = {
        "MAE": mae_test,
        "RMSE": rmse_test,
        "R2_score": r2_test_score
    }
    
    # Affichage des résultats
    print("**** Train Result *****")
    print(result_model_train)
    print("\n**** Test Result *****")
    print(result_model_test)
    
    return result_model_train, result_model_test

# 3. Sauvegarde du pipeline
def save_pipeline(transformer, filename='pipeline.pkl'):
    """Sauvegarder le transformer (pipeline) dans un fichier."""
    joblib.dump(transformer, filename)

# 4. Sauvegarde du modèle
def save_model(model, filename='model_best.pkl'):
    """Sauvegarder le modèle dans un fichier."""
    joblib.dump(model, filename)

# 5. Prédiction avec un modèle et de nouvelles données
def make_prediction(model, transformer, test_data):
    """Appliquer le modèle sur de nouvelles données transformées."""
    
    # Transformer les nouvelles données
    df_test = pd.DataFrame([test_data], columns=X.columns)
    transformed_test = transformer.transform(df_test)
    
    # Faire des prédictions
    prediction_test = model.predict(transformed_test)
    
    return prediction_test

# Exemple de script d'entraînement et de sauvegarde du modèle
def main(data, test_data):
    """Script principal pour entraîner, évaluer, et sauvegarder le modèle."""
    
    # Étape 1: Transformation des données
    Xtrain_new, Xtest_new, ytrain, ytest, transformer = transformation_data(data)
    
    # Étape 2: Entraînement du modèle
    model_best = HuberRegressor(epsilon=1.35, max_iter=200, alpha=0.001)
    train_and_evaluate_model(model_best, Xtrain_new, Xtest_new, ytrain, ytest)
    
    # Étape 3: Sauvegarde du pipeline
    save_pipeline(transformer)
    
    # Étape 4: Sauvegarde du modèle
    save_model(model_best)
    
    # Étape 5: Prédiction avec de nouvelles données
    prediction = make_prediction(model_best, transformer, test_data)
    print(f"Prediction for test data: {prediction}")

# Exemple d'utilisation
test_data = {
    "country": "France",
    "Year": 2023,
    "Region": "Africa",
    "Average CO2 emission factor (tCO2/toe)": 0.5,
    "CO2 intensity at constant purchasing power parities (kCO2/$15p)": 0.2,
    "Total energy production (Mtoe)": 300,
    "Total energy consumption (Mtoe)": 250,
    "Share of renewables in electricity production (%)": 30,
    "Share of electricity in total final energy consumption (%)": 40,
    "Oil products domestic consumption (Mt)": 100,
    "Refined oil products production (Mt)": 80,
    "Natural gas production (bcm)": 50,
    "Natural gas domestic consumption (bcm)": 60,
    "Energy intensity of GDP at constant purchasing power parities (koe/$15p)": 0.4,
    "Electricity production (TWh)": 0.5,
    "Electricity domestic consumption (TWh)": 300,
    "Coal and lignite domestic consumption (Mt)": 150,
    "Share of wind and solar in electricity production (%)": 20,
    "Crude oil production (Mt)": 120,
    "Coal and lignite production (Mt)": 130
}


main(data, test_data)
