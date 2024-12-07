from fastapi import FastAPI, HTTPException
import joblib, pickle
import pandas as pd
import requests, os, pathlib


model = joblib.load('/home/dona-erick/Projet CO2/PredictCO2/Data_energy_Trend/models/model_best.pkl')
pipeline_data = joblib.load('/home/dona-erick/Projet CO2/PredictCO2/Data_energy_Trend/models/pipeline.pkl')
main = FastAPI()


@main.post('/predict')

def predict(data: dict):
    
    try:
        df = pd.DataFrame([data])
        
        transformed_data = pipeline_data.transform(df)
        
        predictions = model.predict(transformed_data)
        
        return {"predictions:", predictions.tolist()}
    except Exception as e:
        return HTTPException(status_code=404, detail = {str(e)})
        