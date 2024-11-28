import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import streamlit.components.v1 as com



df = pd.read_csv("loneliness.csv")
model = joblib.load('final_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
df_schema = joblib.load('df_schema.pkl')

# Streamlit app
st.markdown("<h1 style='text-align: center; color: #1A3F5A;'><u>The Canadian Loneliness Analysis</u></h1>", unsafe_allow_html=True)
st.write('"The worst part of holding the memories is not the pain. It is the loneliness of it. Memories need to be shared."')
st.write('― Lois Lowry, The Giver, 1993')


st.write("Loneliness can leave people feeling isolated and disconnected from others. It is a complex state of mind that can be caused by life changes, mental health conditions, poor self-esteem, and personality traits.")
st.write("Loneliness can also have serious health consequences including decreased mental wellness and physical problems. Loneliness can have a serious effect on your health, so it is important to be able to recognize signs that you are feeling lonely. It is also important to remember that being alone isn't the same as being lonely.")
st.write("Talk to someone you can trust. Reaching out to someone in your life to talk about what you are feeling is important. This can be someone you know such as a family member, but you might also consider talking to your doctor or a therapist.")
html_snippet = """
<div style="width: 100%; height: 300px;">
    <iframe src="https://lottie.host/embed/82421c54-2ba0-4f5b-b66a-8907db5ab17a/uaI1wP5sPk.lottie" width="100%" height="100%" frameborder="0" allowfullscreen></iframe>
</div>
""" 
st.components.v1.html(html_snippet, height=250)

st.markdown("<h1 style='text-align: center; color: #1A3F5A;'><u>Analyze</u></h1>", unsafe_allow_html=True)
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


st.subheader("You are not alone!")
com.iframe("https://lottie.host/embed/166ff794-5cd7-4b61-aafa-ec7ef2247348/QnG5uWmX45.lottie")
st.write("Thank you!")

# External Resources
st.markdown("[Learn more about loneliness and mental health](https://www.britannica.com/science/loneliness)")


