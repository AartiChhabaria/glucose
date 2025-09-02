import joblib
import numpy as np
import pandas as pd

print("🔄 Loading model...")
# Load saved AI model
model = joblib.load("glucose_model.pkl")

# 🔹 Google Sheet CSV link
sheet_url = "https://docs.google.com/spreadsheets/d/1hoNuXaW_y8QPL3Cb8rhUa3ajGnoeKVEfTRVK8OJ1stI/gviz/tq?tqx=out:csv&sheet=Sheet1"

# 🔹 Features required by the model
features = ["AC_Red", "DC_Red", "AC_IR", "DC_IR", "Heart_Rate", "SpO2", "Perfusion_Index"]

print("📥 Fetching latest data from Google Sheets...")

try:
    # Read the Google Sheet
    df = pd.read_csv(sheet_url)

    # Show the last few rows
    print("\n🔎 Latest data in sheet:")
    print(df.tail())

    # Ensure all required columns exist
    if all(col in df.columns for col in features):
        # Get the latest row
        latest = df[features].iloc[-1].values.reshape(1, -1)

        # Predict glucose
        prediction = model.predict(latest)[0]
        print("\n✅ Predicted Glucose (mg/dL):", round(float(prediction), 2))
    else:
        print(f"❌ Missing required columns in sheet. Expected: {features}")

except Exception as e:
    print(f"⚠️ Error fetching data: {e}")
