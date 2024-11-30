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


# Streamlit app
st.markdown("<h1 style='text-align: center; color: #9EB1EF;'><u>The Canadian Loneliness Analysis</u></h1>", unsafe_allow_html=True)
st.write('"The worst part of holding the memories is not the pain. It is the loneliness of it. Memories need to be shared."')
st.write('― Lois Lowry, The Giver, 1993')


st.write("Loneliness can leave people feeling isolated and disconnected from others. It is a complex state of mind that can be caused by life changes, mental health conditions, poor self-esteem, and personality traits.")
st.write("Loneliness can also have serious health consequences including decreased mental wellness and physical problems. Loneliness can have a serious effect on your health, so it is important to be able to recognize signs that you are feeling lonely. It is also important to remember that being alone isn't the same as being lonely.")
st.write("Talk to someone you can trust. Reaching out to someone in your life to talk about what you are feeling is important. This can be someone you know such as a family member, but you might also consider talking to your doctor or a therapist.")