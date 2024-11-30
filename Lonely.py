import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit.components.v1 as com
import joblib

# Load necessary files
df = pd.read_csv("loneliness.csv")
pipeline = joblib.load('final_pipeline.joblib')
label_encoder = joblib.load('label_encoder.joblib')
df_schema = joblib.load('df_schema.joblib')

# Streamlit app
st.markdown("<h1 style='text-align: center; color: #9EB1EF;'><u>The Canadian Loneliness Analysis</u></h1>", unsafe_allow_html=True)
st.write('"The worst part of holding the memories is not the pain. It is the loneliness of it. Memories need to be shared."')
st.write('― Lois Lowry, The Giver, 1993')

st.write("Loneliness can leave people feeling isolated and disconnected from others. It is a complex state of mind that can be caused by life changes, mental health conditions, poor self-esteem, and personality traits.")
st.write("Loneliness can also have serious health consequences including decreased mental wellness and physical problems. Loneliness can have a serious effect on your health, so it is important to be able to recognize signs that you are feeling lonely. It is also important to remember that being alone isn't the same as being lonely.")
st.write("Talk to someone you can trust. Reaching out to someone in your life to talk about what you are feeling is important. This can be someone you know such as a family member, but you might also consider talking to your doctor or a therapist.")

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

# Use the pipeline to scale and predict
if st.button('Predict'):
    predicted_probabilities = pipeline.predict_proba(input_df)[0]
    predicted_bin = pipeline.predict(input_df)[0]
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

grouped_df = df.groupby('Characteristics')['Indicators'].value_counts().unstack().fillna(0)
grouped_df['Loneliness_Rate'] = grouped_df['Yes, feels lonely'] / grouped_df.sum(axis=1) * 100
grouped_df_sorted = grouped_df.sort_values('Loneliness_Rate', ascending=True)

# Interactive plot using Plotly
fig = px.bar(
    grouped_df_sorted.reset_index(),
    x='Characteristics',
    y='Loneliness_Rate',
    title='Loneliness Rate by Characteristics',
    labels={'Loneliness_Rate': 'Loneliness Rate (%)'},
)
st.plotly_chart(fig)

html_code = """
<div style="text-align: center;">
    <img src="https://www150.statcan.gc.ca/pub/11-627-m/2021090/11-627-m2021090-eng.jpg" width="700">
    <p><small>The loneliness in Canada</small></p>
</div>
"""

st.markdown(html_code, unsafe_allow_html=True)

st.subheader("You are not alone!")
com.iframe("https://lottie.host/embed/166ff794-5cd7-4b61-aafa-ec7ef2247348/QnG5uWmX45.lottie", width=800, height=300)

# External Resources
st.markdown("[Click here to learn more about loneliness and mental health](https://www.britannica.com/science/loneliness)")
