# Laptop_Price_Prediction_deploy
Deploy the model in web Streamlit
Laptop Price Prediction
This repository contains a machine learning project that predicts laptop prices based on various features. The project involves data preprocessing, model training, model saving, and deploying the model as a web application using Streamlit.

#Project Overview
1.Data Preprocessing\
2.Model Training\
3.Model Saving\
#Web Deployment with Streamlit\
1. Data Preprocessing\
The first step is to preprocess the data. This includes handling missing values, encoding categorical variables, and scaling numerical features.\

##Preprocessing Script\

import pickle\
pickle.dump(preprocessor, open('preprocessor.pkl', 'wb'))\
2. Model Training\
Train a machine learning model using the preprocessed data.\

#Training Script\
from sklearn.ensemble import RandomForestRegressor\
from sklearn.pipeline import Pipeline

# Load preprocessor
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create and train the pipeline
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('model', model)])

# Split data
X = df.drop('price', axis=1)\
y = df['price']

from sklearn.model_selection import train_test_split\
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipe.fit(X_train, y_train)

# Save the trained model
pickle.dump(pipe, open('pipe.pkl', 'wb'))
3. Model Saving
The trained model and preprocessor are saved as pipe.pkl using the pickle module.

4. Web Deployment with Streamlit\
Create a web interface to interact with the model using Streamlit.

Streamlit App Script\
import streamlit as st\
import pickle\
import numpy as np\

# Load model
pipe = pickle.load(open('pipe.pkl', 'rb'))\
df = pickle.load(open('df.pkl', 'rb'))\

st.title('Laptop Price Predictor')

# Input fields
company = st.selectbox('Brand', df['Company'].unique())
type_ = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM(GB)', [2, 4, 6, 8, 12, 16, 32])
weight = st.number_input('Weight of Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['Yes', 'No'])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 128, 256, 512, 1024, 2048])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Laptop Price'):
    # Convert inputs to numerical format
    touchscreen = 1 if touchscreen == 'Yes' else 0\
    ips = 1 if ips == 'Yes' else 0\
    x_res = int(resolution.split('x')[0])\
    y_res = int(resolution.split('x')[1])\
    ppi = ((x_res**2) + (y_res**2))**0.5 / screen_size\

    # Create input array for the model
    query = np.array([company, type_, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)

    # Predict and display the price
    prediction = pipe.predict(query)
    st.title(f"Predicted Laptop Price: {int(np.exp(prediction[0]))}")
Running the App
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/laptop-price-prediction.git
cd laptop-price-prediction
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run app.py
##Conclusion:-
This project demonstrates a complete workflow from data preprocessing, model training, saving the model, and deploying it using Streamlit for laptop price prediction. Contributions and improvements are welcome!
