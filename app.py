import os
import requests
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Telco Churn Predictor', layout='wide')
st.title('Telco Customer Churn Predictor')

# Get backend URL from environment variables, defaulting to local FastAPI dev server
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
if not BACKEND_URL.startswith(("http://", "https://")):
    BACKEND_URL = f"https://{BACKEND_URL}"

# Load selected features list and importance data for UI
selected_features = joblib.load('models/selected_features.pkl')

st.sidebar.header('Customer details')

# Collect user inputs
input_data = {}
for feature in selected_features:
    if feature == 'tenure':
        input_data[feature] = st.sidebar.slider('Tenure', 0, 72, 12)
    elif feature == 'MonthlyCharges':
        input_data[feature] = st.sidebar.number_input('Monthly charges', 0.0, 200.0, 70.0)
    elif feature == 'TotalCharges':
        input_data[feature] = st.sidebar.number_input('Total charges', 0.0, 10000.0, 1000.0)
    elif feature == 'AvgChargesPerTenure':
        input_data[feature] = st.sidebar.number_input('Average charges per tenure', 0.0, 300.0, 70.0)
    elif feature == 'SeniorCitizen':
        input_data[feature] = st.sidebar.selectbox('Senior citizen', [0, 1])
    elif feature == 'IsMonthToMonth':
        input_data[feature] = st.sidebar.selectbox('Month-to-month contract?', [0, 1])
    elif feature == 'HasFiberOptic':
        input_data[feature] = st.sidebar.selectbox('Has fiber optic?', [0, 1])
    elif feature == 'Contract':
        input_data[feature] = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    elif feature == 'PaymentMethod':
        input_data[feature] = st.sidebar.selectbox('Payment method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    elif feature == 'OnlineSecurity':
        input_data[feature] = st.sidebar.selectbox('Online security', ['No', 'Yes', 'No internet service'])
    elif feature == 'TechSupport':
        input_data[feature] = st.sidebar.selectbox('Tech support', ['No', 'Yes', 'No internet service'])
    elif feature == 'InternetService':
        input_data[feature] = st.sidebar.selectbox('Internet service', ['DSL', 'Fiber optic', 'No'])
    elif feature == 'OnlineBackup':
        input_data[feature] = st.sidebar.selectbox('Online backup', ['No', 'Yes', 'No internet service'])
    elif feature == 'PaperlessBilling':
        input_data[feature] = st.sidebar.selectbox('Paperless billing', ['Yes', 'No'])
    else:
        input_data[feature] = st.sidebar.text_input(feature, '')

st.subheader('Prediction')
if st.button("Predict Churn"):
    try:
        api_endpoint = f"{BACKEND_URL}/predict"
        with st.spinner("Sending request to backend prediction service..."):
            response = requests.post(api_endpoint, json=input_data, timeout=10)
            
        if response.status_code == 200:
            result = response.json()
            probability = result["churn_probability"]
            prediction = result["prediction"]
            
            st.metric('Churn probability', f'{probability:.1%}')
            st.write(f'Final prediction: **{prediction}**')
        else:
            st.error(f"Backend API returned an error ({response.status_code}): {response.text}")
            
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the backend prediction API at {api_endpoint}. Is the server running? Error: {e}")
        
st.write("---")
st.dataframe(pd.DataFrame([input_data]), use_container_width=True)
