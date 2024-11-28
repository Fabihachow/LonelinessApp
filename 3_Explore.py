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

st.markdown("<h1 style='text-align: center; color: #9EB1EF;'><u>Visualisations</u></h1>", unsafe_allow_html=True)


# Interactive scatter plot
fig = px.scatter(df, x='REF_DATE', y='VALUE', color='GEO', title='Loneliness Percentage Over the Years by Region', 
                 labels={"REF_DATE": "Year", "VALUE": "Percentage %", "GEO": "Region"}, 
                 hover_data=["REF_DATE", "VALUE", "GEO"])
st.plotly_chart(fig)

# Interactive bar plot
fig = px.box(df, x='Age group', y='VALUE', title='Loneliness Percentage by Age Group',
             labels={"Age group": "Age Groups", "VALUE": "Percentage %"},
             color='Age group')


st.plotly_chart(fig)


html_code = """
<div style="text-align: center;">
    <img src="https://www150.statcan.gc.ca/pub/11-627-m/2021090/11-627-m2021090-eng.jpg" width="700">
    <p><small>The loneliness in Canada</small></p>
</div>
"""

st.markdown(html_code, unsafe_allow_html=True)


st.subheader("You are not alone!")
com.iframe("https://lottie.host/embed/166ff794-5cd7-4b61-aafa-ec7ef2247348/QnG5uWmX45.lottie", width=900, height=300)



# External Resources
st.markdown("[Click here to learn more about loneliness and mental health](https://www.britannica.com/science/loneliness)")
st.markdown("<small>Fabiha Chowdhury 2024</small>", unsafe_allow_html=True)