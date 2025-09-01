import streamlit as st
import joblib
import numpy as np

st.title("🩺 AI Glucose Monitor")

# User inputs using sliders
optical = st.slider("Optical Value", 400, 800, 500)
impedance = st.slider("Impedance", 80, 150, 100)
pulse = st.slider("Pulse", 60, 100, 75)
temperature = st.slider("Temperature (°C)", 35.0, 38.0, 36.5)

# Load trained model
model = joblib.load("glucose_model.pkl")

# Predict
new_data = np.array([[optical, impedance, pulse, temperature]])
prediction = model.predict(new_data)

st.subheader(f"Predicted Glucose: {round(prediction[0], 2)} mg/dL")
