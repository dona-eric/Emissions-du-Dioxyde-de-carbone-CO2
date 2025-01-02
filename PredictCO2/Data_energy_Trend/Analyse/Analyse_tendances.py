#!/usr/bin/env python
# coding: utf-8

# Importations des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Suppression des warnings
import warnings
warnings.filterwarnings(action='ignore')


def load_data(path):
    """Charge les données à partir d'un fichier CSV."""
    return pd.read_csv(path)


def plot_distribution(data):
    """Affiche la distribution des variables catégorielles et numériques."""
    # Graphiques pour variables catégorielles
    for var in data.select_dtypes(exclude=['int64', 'float64']).columns:
        plt.figure(figsize=(10, 8))
        sns.countplot(data=data, y=var)
        plt.title(f'Distribution par {var}')
        plt.tight_layout()
        plt.show()

    # Boxplots pour variables numériques
    numeric_cols = data.select_dtypes(include=['float64', 'int64'])
    for col in numeric_cols.columns:
        plt.figure(figsize=(10, 8))
        sns.boxplot(data=data, y=col, palette='Set2')
        plt.title(f'Distribution de {col}')
        plt.tight_layout()
        plt.show()


def visualize_correlation(data):
    """Affiche la matrice de corrélation des variables numériques."""
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matrice de corrélation des variables numériques')
    plt.show()


def transformation_data(data):
    """Prépare les données pour l'entraînement en appliquant les transformations nécessaires."""
    X = data.drop(columns=['CO2 emissions from fuel combustion (MtCO2)'])
    y = data['CO2 emissions from fuel combustion (MtCO2)']
    
    # Identifier les variables numériques et catégorielles
    num_var = X.select_dtypes(exclude='object').columns
    cat_var = X.select_dtypes(include='object').columns
    
    # Pipeline pour les variables numériques
    pipeline_numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline pour les variables catégorielles
    pipeline_categorical = Pipeline([
        ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Transformation des colonnes avec ColumnTransformer
    transformer = ColumnTransformer([
        ('num', pipeline_numeric, num_var),
        ('cat', pipeline_categorical, cat_var)
    ])
    
    # Diviser les données en train et test
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=80)
    
    # Appliquer les transformations
    Xtrain_new = transformer.fit_transform(Xtrain)
    Xtest_new = transformer.transform(Xtest)
    
    return Xtrain_new, Xtest_new, ytrain, ytest, transformer


def train_and_evaluate_model(estimator, Xtrain, Xtest, ytrain, ytest):
    """Entraîner un modèle et évaluer ses performances."""
    # Entraîner le modèle
    estimator.fit(Xtrain, ytrain)
    
    # Prédictions et évaluations
    predict_train = estimator.predict(Xtrain)
    predict_test = estimator.predict(Xtest)
    
    results_train = {
        "MAE": mean_absolute_error(ytrain, predict_train),
        "RMSE": np.sqrt(mean_squared_error(ytrain, predict_train)),
        "R2_score": r2_score(ytrain, predict_train)
    }
    
    results_test = {
        "MAE": mean_absolute_error(ytest, predict_test),
        "RMSE": np.sqrt(mean_squared_error(ytest, predict_test)),
        "R2_score": r2_score(ytest, predict_test)
    }
    
    # Affichage des résultats
    print("**** Train Result *****")
    print(results_train)
    print("\n**** Test Result *****")
    print(results_test)
    
    return results_train, results_test


def save_pipeline(transformer, filename='pipeline.pkl'):
    """Sauvegarde le pipeline de transformation des données."""
    joblib.dump(transformer, filename)


def save_model(model, filename='model_best.pkl'):
    """Sauvegarde le modèle entraîné."""
    joblib.dump(model, filename)


def make_prediction(model, transformer, test_data):
    """Applique le modèle sur de nouvelles données transformées."""
    df_test = pd.DataFrame([test_data])
    transformed_test = transformer.transform(df_test)
    prediction_test = model.predict(transformed_test)
    return prediction_test


def main(data, test_data):
    """Script principal pour entraîner, évaluer, et sauvegarder le modèle."""
    # Transformation des données
    Xtrain_new, Xtest_new, ytrain, ytest, transformer = transformation_data(data)
    
    # Entraînement du modèle
    model_best = HuberRegressor(epsilon=1.35, max_iter=200, alpha=0.001)
    train_and_evaluate_model(model_best, Xtrain_new, Xtest_new, ytrain, ytest)
    
    # Sauvegarde du pipeline et du modèle
    save_pipeline(transformer)
    save_model(model_best)
    
    # Prédiction sur de nouvelles données
    prediction = make_prediction(model_best,test_data)
    print(f"Prediction for test data: {prediction}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Charger les données
    data_path = '/home/dona-erick/Projet CO2/PredictCO2/data/Energy data 1990 - 2020.csv'
    data = load_data(data_path)
    
    # Tester la fonction avec des données de test
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
    
    # Appel de la fonction principale
    main(data, test_data)
