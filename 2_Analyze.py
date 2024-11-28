import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit.components.v1 as com
import joblib


df = pd.read_csv("loneliness.csv")
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')
df_schema = joblib.load('df_schema.joblib')
model= joblib.load('final_model.joblib')

st.markdown("<h1 style='text-align: center; color: #9EB1EF;'><u>Analyze</u></h1>", unsafe_allow_html=True)
# Input form
characteristics = [
    'Living with two parents', 'Other', 'Living with lone parent',
    'Not living with any parent', 'Immigrant', 'Native',
    'Visible Minority', 'Retired', 'LGBTQ2+', 'Person with a disability',
    'Non-permanent residents'
]
indicators_options = ['Yes, feels lonely', 'No, do not feel lonely', 'Sometimes, feels lonely']
age_groups = [
    '0 to 4 years', '5 to 10 years', '11 to 14 years', '15 to 19 years',
    '20 to 24 years', '25 to 64 years', '65 + years'
]
provinces = [
    'Territories', 'Canada', 'Ontario', 'Quebec', 'Manitoba', 'Saskatchewan',
    'British Columbia', 'Nova Scotia', 'New Brunswick', 'Alberta', 'Newfoundland and Labrador', 'Prince Edward Island'
]
years = list(range(1995, 2025))
gender_options = ['X Gender', 'Woman', 'Man']

# Create selection boxes
selected_characteristic = st.selectbox('Select a Characteristic', characteristics)
selected_indicator = st.selectbox('Select an Indicator', indicators_options)
selected_age_group = st.selectbox('Select an Age Group', age_groups)
selected_province = st.selectbox('Select Province', provinces)
selected_year = st.selectbox('Select Year', years)
selected_gender = st.selectbox('Select Gender', gender_options)

# Prepare the input data
user_input = {
    'REF_DATE': selected_year,
    'GEO': selected_province,
    'Gender': selected_gender,
    'Indicators': selected_indicator,
    'Age group': selected_age_group,
    'Characteristics': selected_characteristic
}

input_df = pd.DataFrame([user_input])
input_df = pd.get_dummies(input_df).reindex(columns=df_schema, fill_value=0)
input_scaled = scaler.transform(input_df)

# Make predictions
if st.button('Predict'):
    predicted_probabilities = model.predict_proba(input_scaled)[0]
    predicted_bin = model.predict(input_scaled)[0]
    predicted_class_name = label_encoder.inverse_transform([predicted_bin])[0]
    predicted_class_probability = predicted_probabilities[predicted_bin] * 100
    st.write(f'Predicted Class: {predicted_class_name}')
    st.write(f'Prediction Percentage: {predicted_class_name}: {predicted_class_probability:.2f}%')

    prob_df = pd.DataFrame({'Class': label_encoder.classes_, 'Probability': predicted_probabilities * 100})
    st.bar_chart(prob_df.set_index('Class'))
    st.subheader("How to Interpret the Bar Chart")
    st.write("""
    The bar chart displays the predicted probabilities for each possible class. Here's what it means:
    
    - **Predicted Probabilities**: Each bar represents the probability percentage for that class as determined by the model.
    - **Class Labels**: The x-axis shows the different classes that the model can predict (e.g., High, Low, Medium).
    - **Probability Values**: The y-axis shows the probability percentage that the model assigns to each class for the given input data.

    ### Interpretation
    - **Highest Bar**: The class with the highest bar is the one that the model is most confident about. This is usually the predicted class that the model outputs.
    - **Other Bars**: The other bars represent the model's confidence in the other classes. Even if one class has the highest probability, the model might still show some confidence in other classes, which is reflected in the heights of these bars.
    """)
    st.image("https://www.meaning.ca/web/wp-content/uploads/2020/10/Picture1.png", caption="The Epidemic of loneliness", use_container_width=True)



