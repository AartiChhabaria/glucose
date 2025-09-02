import streamlit as st
import pandas as pd
import joblib
import time

st.title("🩺 Live AI Glucose Monitor (Google Sheets Input)")

st.markdown("""
This app fetches the latest data from your **Google Sheet** and predicts glucose levels 
using your trained model.
""")

# 🔹 Google Sheet CSV link (your sheet)
sheet_url = "https://docs.google.com/spreadsheets/d/1hoNuXaW_y8QPL3Cb8rhUa3ajGnoeKVEfTRVK8OJ1stI/gviz/tq?tqx=out:csv&sheet=Sheet1"

# 🔹 Load trained model
model = joblib.load("glucose_model.pkl")

# 🔹 Auto-refresh interval
refresh_rate = st.slider("Refresh every (seconds)", 5, 60, 10)

# Placeholder for live updates
placeholder = st.empty()

features = ["AC_Red", "DC_Red", "AC_IR", "DC_IR", "Heart_Rate", "SpO2", "Perfusion_Index"]

while True:
    try:
        # Read live sheet
        df = pd.read_csv(sheet_url)

        with placeholder.container():
            st.write("📥 Latest data from Google Sheets:")
            st.dataframe(df.tail())

            if all(col in df.columns for col in features):
                latest = df[features].iloc[-1].values.reshape(1, -1)
                prediction = model.predict(latest)[0]
                st.success(f"Predicted Glucose: **{round(float(prediction), 2)} mg/dL**")
            else:
                st.error(f"❌ Sheet must have columns: {features}")

    except Exception as e:
        st.error(f"⚠️ Could not fetch data: {e}")

    time.sleep(refresh_rate)
