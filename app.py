import numpy as np
import pickle
import streamlit as st
import base64

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]

def set_background():
    with open("GettyImages-1011263454.jpg", "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64_encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        header[data-testid="stHeader"] {{
            display: none;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    set_background()
    st.title('Heart Disease Prediction')
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=50)
        sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
        cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
        trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
        chol = st.number_input('Cholesterol', min_value=100, max_value=600, value=200)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        restecg = st.selectbox('Resting ECG', options=[0, 1, 2])
    
    with col2:
        thalach = st.number_input('Max Heart Rate', min_value=60, max_value=220, value=150)
        exang = st.selectbox('Exercise Induced Angina', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        oldpeak = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox('Slope of Peak Exercise ST', options=[0, 1, 2])
        ca = st.selectbox('Number of Major Vessels', options=[0, 1, 2, 3, 4])
        thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3])
    
    if st.button('Predict Heart Disease'):
        diagnosis = heart_disease_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        
        if diagnosis == 1:
            st.error('⚠️ The person has heart disease')
        else:
            st.success('✅ The person does not have heart disease')

if __name__ == '__main__':
    main()