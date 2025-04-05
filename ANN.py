import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import h5py 

# Load the trained model
model = tf.keras.models.load_model(r'E:\Desktop\sadia waheed GU1\Gen AI course 4 months\ANN-P2\env\model.h5') 

# Load the encoder and scaler
with open('E:\Desktop\sadia waheed GU1\Gen AI course 4 months\ANN-P2\env\label_gender.pkl', 'rb') as file:
    label_gender = pickle.load(file)
with open('E:\Desktop\sadia waheed GU1\Gen AI course 4 months\ANN-P2\env\one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)
with open('E:\Desktop\sadia waheed GU1\Gen AI course 4 months\ANN-P2\env\scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Predicted Salary")

# User input
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number Of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited = st.selectbox('Exited', [0, 1])

# One-hot encode the geography
gender_encoded = label_gender.transform([gender])[0] 
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))


# Prepare the input data for prediction
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})


# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

from sklearn.preprocessing import StandardScaler
import pandas as pd

import numpy as np
import tensorflow as tf


# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict Salary using the trained model
prediction = model.predict(input_data_scaled)

# Get the predicted salary from the model's output
predicted_salary = prediction[0][0]

# Display the predicted salary
st.write(f"Predicted Salary: ${predicted_salary:.2f}")



