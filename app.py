import streamlit as st # membuat web app interaktif
import joblib # load model dan scaler yang sudah disimpan
import numpy as np
import pandas as pd

# Load preprocess and model from MLflow
# Load preprocessor
scaler = joblib.load("preprocessor.pkl") # scaler hasil preprocessing
model = joblib.load("model.pkl") # model hasil training

continuous_cols = ["age","trestbps","chol","thalach","oldpeak"]

def main():
    st.title('Machine Learning Heart Attack Model Deployment')
    st.write("Masukkan data pasien untuk memprediksi risiko serangan jantung")

    # Add user input components for 5 features
    # jangan lupa set nilai min dan max agar invalid data tidak masuk
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Sex", [0,1])
    st.caption("0 = Female, 1 = Male")
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
    restecg = st.selectbox("Resting ECG Result", [0,1,2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", [0,1])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0,1,2])
    ca = st.selectbox("Number of Major Vessels", [0,1,2,3])
    thal = st.selectbox("Thalassemia", [0,1,2,3])
    
    # prediksi, ketika tombol ditekan ambil semua fitru, masukkan ke fungsi make_prediction(), tampilkan hasil 
    if st.button('Make Prediction'):
        features = [age,sex,cp,trestbps,chol,fbs,
            restecg,thalach,exang,oldpeak,
            slope,ca,thal] # mengumpulkan fitur
        result = make_prediction(features) # ubah input jadi array, scaling, prediksi, return hasil 
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    columns = [
        "age","sex","cp","trestbps","chol","fbs",
        "restecg","thalach","exang","oldpeak",
        "slope","ca","thal"
    ]

    input_df = pd.DataFrame([features], columns=columns)

    # scaling hanya untuk fitur kontinu
    input_df[continuous_cols] = scaler.transform(input_df[continuous_cols])

    prediction = model.predict(input_df)

    return prediction[0]
    prediction = model.predict(X_scaled)
    return prediction[0]

if __name__ == '__main__':
    main()


