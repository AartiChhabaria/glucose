import joblib
import numpy as np

print("🔄 Loading model and making prediction...")
# Load saved AI model
model = joblib.load("glucose_model.pkl")

# Example input (optical, impedance, pulse, temperature)
new_data = np.array([[520, 105, 80, 36.7]])

# Predict glucose
prediction = model.predict(new_data)

print("Predicted Glucose (mg/dL):", round(prediction[0], 2))
