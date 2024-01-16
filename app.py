import numpy as np
import pickle
import streamlit as st
import pandas as pd

# Load the trained model
with open('best_dt_model_finalpro.predict.pkl', 'rb') as f:
    xgr_model = pickle.load(f)

# Placeholder encoding mappings (replace with your actual mappings)
encoding_col_type = {'A': 1, 'B': 2, 'C': 3}
encoding_col_hol = {True: 1, False: 2}  # Assuming 'IsHoliday_y' is boolean

# Load your existing DataFrame
# Replace 'your_dataframe.csv' with the actual path or filename of your DataFrame CSV file
your_dataframe = pd.read_csv('retail_clean.csv')

# Streamlit App
st.set_page_config(
    page_title="Retail Store Weekly Sales Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title(":red[Retail store] :blue[weekly sales] :orange[Prediction]")

# Columns for User Input
col1, col2 = st.columns(2, gap='large')

with col1:
    store = st.slider("Select Store", min_value=1, max_value=45, value=1, step=1)
    type_input = st.selectbox("Select Type For your info: 1 = A - type, 2 = B - Type, 3 = C - type", your_dataframe['Type'].unique(), key="select_type")
    size = st.number_input("Enter Size", min_value=your_dataframe['Size'].min(), max_value=your_dataframe['Size'].max(), value=int(your_dataframe['Size'].mean()))
    dept = st.number_input("Enter Dept (Max- 45)", min_value=your_dataframe['Dept'].min(), max_value=your_dataframe['Dept'].max(), value=int(your_dataframe['Dept'].mean()))
    year = st.number_input("Enter Year", min_value=your_dataframe['Year'].min(), max_value=your_dataframe['Year'].max(), value=int(your_dataframe['Year'].mean()))
    month = st.slider("Select Month", min_value=1, max_value=12, value=1, step=1)
    week = st.number_input("Enter Week", min_value=your_dataframe['Week'].min(), max_value=your_dataframe['Week'].max(), value=int(your_dataframe['Week'].mean()))
    temperature = st.number_input("Enter Temperature", min_value=your_dataframe['Temperature'].min(), max_value=your_dataframe['Temperature'].max(), value=your_dataframe['Temperature'].mean())
    fuel_price = st.number_input("Enter Fuel Price", min_value=your_dataframe['Fuel_Price'].min(), max_value=your_dataframe['Fuel_Price'].max(), value=your_dataframe['Fuel_Price'].mean())

# User Input Section - Right Column
with col2:
    markdown1 = st.number_input("Enter MarkDown1", min_value=your_dataframe['MarkDown1'].min(), max_value=your_dataframe['MarkDown1'].max(), value=your_dataframe['MarkDown1'].mean())
    markdown2 = st.number_input("Enter MarkDown2", min_value=your_dataframe['MarkDown2'].min(), max_value=your_dataframe['MarkDown2'].max(), value=your_dataframe['MarkDown2'].mean())
    markdown3 = st.number_input("Enter MarkDown3", min_value=your_dataframe['MarkDown3'].min(), max_value=your_dataframe['MarkDown3'].max(), value=your_dataframe['MarkDown3'].mean())
    markdown4 = st.number_input("Enter MarkDown4", min_value=your_dataframe['MarkDown4'].min(), max_value=your_dataframe['MarkDown4'].max(), value=your_dataframe['MarkDown4'].mean())
    markdown5 = st.number_input("Enter MarkDown5", min_value=your_dataframe['MarkDown5'].min(), max_value=your_dataframe['MarkDown5'].max(), value=your_dataframe['MarkDown5'].mean())
    cpi = st.number_input("Enter CPI", min_value=your_dataframe['CPI'].min(), max_value=your_dataframe['CPI'].max(), value=your_dataframe['CPI'].mean())
    unemployment = st.number_input("Enter Unemployment", min_value=your_dataframe['Unemployment'].min(), max_value=your_dataframe['Unemployment'].max(), value=your_dataframe['Unemployment'].mean())
    is_holiday = st.checkbox("Is Holiday")

# Predictions Section
st.title("Predictions")
prediction_button = st.button('Predict Weekly Sales')

# Check if the prediction button is clicked
if prediction_button:
    # Encode 'Type' input
    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        'Store': [store],
        'Type': [type_input],
        'Size': [size],
        'Dept': [dept],
        'Year': [year],
        'Month': [month],
        'Week': [week],
        'Temperature': [temperature],
        'Fuel_Price': [fuel_price],
        'MarkDown1': [markdown1],
        'MarkDown2': [markdown2],
        'MarkDown3': [markdown3],
        'MarkDown4': [markdown4],
        'MarkDown5': [markdown5],
        'CPI': [cpi],
        'Unemployment': [unemployment],
        'IsHoliday_y': [is_holiday]
    })

    # Make predictions
    predicted_sales = xgr_model.predict(user_input)

    # Display the prediction
    st.write(f"Predicted Weekly Sales: {predicted_sales[0]}")
