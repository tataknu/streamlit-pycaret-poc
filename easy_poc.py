from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


FILENAME = "results.csv"
MODEL_RESULTS = "/Users/cristianvildosola/Develop/streamlit/"
TRAIN_DATA = "/Users/cristianvildosola/Downloads/data.csv"

def predict_score(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    return predictions_data['Label'][0]
    
model = load_model('blended-model-01')


st.sidebar.title("Input variables for prediction:")

st.title('Credit Score Prediction')
st.write('This is a web app to calculate your credit score based on personal data.')

st.markdown('#')


age = st.sidebar.slider(label = 'Age', min_value = 18,
                          max_value = 99 ,
                          value = 18,
                          step = 1)

salary = st.sidebar.slider(label = 'Salary', min_value = 10000,
                          max_value = 200000 ,
                          value = 10000,
                          step = 1000)

balance = st.sidebar.slider(label = 'Total Balance', min_value = 0,
                          max_value = 300000 ,
                          value = 1000,
                          step = 1000)

features = {
    'Age': age, 
    'Salary': salary,
    'Balance': balance,
            }

features_df  = pd.DataFrame([features])
st.table(features_df[['Age', 'Salary', 'Balance',]]) 

if st.button('Predict'):
    
    prediction = predict_score(model, features_df)
    
    st.write(' Based on your personal information your credit score is: '+ str(prediction))


# Model comparision
results = pd.read_csv(MODEL_RESULTS+FILENAME)
st.dataframe(results)

st.markdown('#')
st.write('Score by age')
df = pd.read_csv(TRAIN_DATA)
df = df[['Age', 'Score']].groupby('Age')[['Score']].mean()
st.line_chart(df)