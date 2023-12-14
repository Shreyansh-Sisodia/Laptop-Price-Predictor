# Import necessary libraries
import streamlit as st
import pickle
import numpy as np

# Load the pre-trained machine learning model and data
with open('pipeline.pkl', 'rb') as file:
    # Load the pre-trained machine learning model
    pipe = pickle.load(file)
with open('laptop.pkl', 'rb') as file:
    # Load the dataset used for the model
    df = pickle.load(file)

# Set the title of the Streamlit web app
st.title("Laptop Price Predictor")

# User inputs for various laptop features
company = st.selectbox('Company', df['Company'].unique())
laptop_type = st.selectbox('Model', df['TypeName'].unique())
os = st.selectbox('Operating System', df['os'].unique())
screen_size = st.slider('Display Dimensions:', 10, 30, 15)
weight = st.slider('Mass:', 0.0, 10.0, 5.0)
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
                                                '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
ips = st.radio('IPS Display', ['No', 'Yes'])
cpu = st.selectbox('Processor', df['Cpu brand'].unique())
gpu = st.selectbox('Graphic Card', df['Gpu brand'].unique())
touchscreen = st.radio('Touchscreen', ['No', 'Yes'])
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# Estimate Laptop Price on button click
if st.button('Estimate Laptop Price'):
    # Calculate pixels per inch (ppi) using tuple unpacking
    res_x, res_y = map(int, resolution.split('x'))
    ppi = (res_x**2 + res_y**2)**0.5 / screen_size

    # Convert touchscreen and IPS options to binary values using map
    touchscreen_binary, ips_binary = map(lambda x: 1 if x == 'Yes' else 0, [touchscreen, ips])

    # Create a numpy array with the selected features
    query = np.array([company, laptop_type, ram, weight, touchscreen_binary, ips_binary, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, 12)

    # Display the estimated laptop price
    estimated_price = int(np.exp(pipe.predict(query)[0]))
    st.title(f"The estimated cost of this setup is {estimated_price} Indian Rupees")