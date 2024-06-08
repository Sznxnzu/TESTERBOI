import streamlit as st
import joblib
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

model1 = joblib.load('model_Indonesia.pkl')
model2 = joblib.load('model_China.pkl')

def make_predictions_and_plot(data,model):
    predictions = model.predict(data)
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, predictions, label='Predictions')
    plt.xlabel('Date')
    plt.ylabel('Prediction')
    plt.title('Model Predictions')
    plt.legend()
    st.pyplot()

def main():
    st.title('Time Series Regression Model Deployment')

    st.header('Select Number of Years:')
    num_years = st.slider('Select number of years', 1, 10, 5)

    model_selection = st.selectbox('Select Model', ['Model 1', 'Model 2'])

    if model_selection == 'Model 1':
        model = model1
    elif model_selection == 'Model 2':
        model = model2
        
    make_predictions_and_plot(num_years,model)

if __name__ == '__main__':
    main()