import streamlit as st
import pandas as pd
import requests

url_api = 'http://127.0.0.1:8000/predict'

st.title('Prediction de CO2')

data = {}

data['country'] = st.text_input('Entrer un pays de votre choix:')
data['Year']= st.number_input('Entrer une année', max_value=2024, min_value=1990)
data['Region']= st.text_input('Entrer la region')
data['Average CO2 emission factor (tCO2/toe)'] = st.number_input('Entrer CAverage CO2 emission factor (tCO2/toe)')
data['CO2 intensity at constant purchasing power parities (kCO2/$15p)'] = st.number_input('Entrer CO2 intensity at constant purchasing power parities (kCO2/$15p)')
data['Total energy production (Mtoe)'] = st.number_input('Entrer Total energy production (Mtoe)')
data['Total energy consumption (Mtoe)'] = st.number_input('Entrer Total energy consumption (Mtoe)')
data['Share of renewables in electricity production (%)'] = st.number_input('Entrer Share of renewables in electricity production (%)')
data['Share of electricity in total final energy consumption (%)'] = st.number_input('Entrer Share of electricity in total final energy consumption (%)	')
data['Oil products domestic consumption (Mt) '] = st.number_input('Entrer Oil products domestic consumption (Mt) ')
data['Refined oil products production (Mt)'] = st.number_input('Entrer  Refined oil products production (Mt)')
data['Natural gas production (bcm) '] = st.number_input('Entrer CNatural gas production (bcm) ')
data['Natural gas domestic consumption (bcm)'] = st.number_input('Entrer  Natural gas domestic consumption (bcm)')
data['Energy intensity of GDP at constant purchasing power parities (koe/$15p)'] = st.number_input('Entrer CEnergy intensity of GDP at constant purchasing power parities (koe/$15p)')
data['Electricity domestic consumption (TWh)'] = st.number_input('Entrer Electricity domestic consumption (TWh) ')
data['Coal and lignite domestic consumption (Mt)'] = st.number_input('Entrer Coal and lignite domestic consumption (Mt)')
data['Share of wind and solar in electricity production (%)'] = st.number_input('Entrer Share of wind and solar in electricity production (%)')
data['Crude oil production (Mt'] = st.number_input('Entrer Crude oil production (Mt')
data['Coal and lignite production (Mt)'] = st.number_input('Entrer Coal and lignite production (Mt)')

if st.button('Predire'):
    
    try:
        # appel à l'api
        responses = requests.post(url=url_api, json=data)
        if responses.status_code==200:
            prediction = responses.json()['predictions']
            st.success(f'Prediction :{prediction}')
        else:
            st.write("Erreur ! veuillez ressayer")
    except Exception as e:
        st.error(f"Erreur lors de la requetes vers l'api {str(e)}")

