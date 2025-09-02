import streamlit as st
import pandas as pd
import joblib
import time

st.title("🩺 Live AI Glucose Monitor (Google Sheets Input)")

st.markdown("""
This app fetches the latest data from your **Google Sheet** and predicts glucose levels 
using your trained model.
""")

# 🔹 Google Sheet CSV link
sheet_url = "https://docs.google.com/spreadsheets/d/1hoNuXaW_y8QPL3Cb8rhUa3ajGnoeKVEfTRVK8OJ1stI/gviz/tq?tqx=out:csv&sheet=Sheet1"

# 🔹 Load trained model
model = joblib.load("glucose_model.pkl")

# 🔹 Columns required
features = ["AC_Red", "DC_Red", "AC_IR", "DC_IR", "Heart_Rate", "SpO2", "Perfusion_Index"]

# Placeholder for live updates
placeholder = st.empty()

while True:
    try:
        # Read live sheet
        df = pd.read_csv(sheet_url)

        with placeholder.container():
            st.write("📥 Latest data from Google Sheets:")
            st.dataframe(df.tail())

            # ✅ Get refresh interval from sheet (must have column 'Refresh_Rate')
            if "Refresh_Rate" in df.columns:
                refresh_rate = int(df["Refresh_Rate"].iloc[-1])  # last row's refresh value
            else:
                refresh_rate = 10  # default fallback

            # ✅ Prediction
            if all(col in df.columns for col in features):
                latest = df[features].iloc[-1].values.reshape(1, -1)
                prediction = model.predict(latest)[0]
                st.success(f"Predicted Glucose: **{round(float(prediction), 2)} mg/dL**")
            else:
                st.error(f"❌ Sheet must have columns: {features}")

            st.info(f"⏳ Auto-refreshing every {refresh_rate} seconds (from sheet).")

    except Exception as e:
        st.error(f"⚠️ Could not fetch data: {e}")

    time.sleep(refresh_rate)
