# import streamlit as st
# import pandas as pd
# import joblib
# import time

# st.title("🩺 Live AI Glucose Monitor (Google Sheets Input)")

# st.markdown("""
# This app fetches the latest data from your **Google Sheet** and predicts glucose levels 
# using your trained model.
# """)

# # 🔹 Google Sheet CSV link
# sheet_url = "https://docs.google.com/spreadsheets/d/1hoNuXaW_y8QPL3Cb8rhUa3ajGnoeKVEfTRVK8OJ1stI/gviz/tq?tqx=out:csv&sheet=Sheet1"

# # 🔹 Load trained model
# model = joblib.load("glucose_model.pkl")

# # 🔹 Columns required
# features = ["AC_Red", "DC_Red", "AC_IR", "DC_IR", "Heart_Rate", "SpO2", "Perfusion_Index"]

# # Placeholder for live updates
# placeholder = st.empty()

# while True:
#     try:
#         # Read live sheet
#         df = pd.read_csv(sheet_url)

#         with placeholder.container():
#             st.write("📥 Latest data from Google Sheets:")
#             st.dataframe(df.tail())

#             # ✅ Get refresh interval from sheet (must have column 'Refresh_Rate')
#             if "Refresh_Rate" in df.columns:
#                 refresh_rate = int(df["Refresh_Rate"].iloc[-1])  # last row's refresh value
#             else:
#                 refresh_rate = 10  # default fallback

#             # ✅ Prediction
#             if all(col in df.columns for col in features):
#                 latest = df[features].iloc[-1].values.reshape(1, -1)
#                 prediction = model.predict(latest)[0]
#                 st.success(f"Predicted Glucose: **{round(float(prediction), 2)} mg/dL**")
#             else:
#                 st.error(f"❌ Sheet must have columns: {features}")

#             st.info(f"⏳ Auto-refreshing every {refresh_rate} seconds (from sheet).")

#     except Exception as e:
#         st.error(f"⚠️ Could not fetch data: {e}")

#     time.sleep(refresh_rate)

import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

st.title("🩺 Live AI Glucose Monitor (Google Sheets Input + Future Prediction)")

st.markdown("""
This app fetches the latest data from your **Google Sheet** and:
- Shows real-time glucose predictions
- Forecasts the next **30–60 minutes**
- Alerts if levels are predicted to go **too high (>180 mg/dL)** or **too low (<70 mg/dL)**
""")

# 🔹 Google Sheet CSV link
sheet_url = "https://docs.google.com/spreadsheets/d/1hoNuXaW_y8QPL3Cb8rhUa3ajGnoeKVEfTRVK8OJ1stI/gviz/tq?tqx=out:csv&sheet=Sheet1"

# 🔹 Load trained model (for instant prediction from sensor data)
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

            # ✅ Get refresh interval from sheet
            if "Refresh_Rate" in df.columns:
                refresh_rate = int(df["Refresh_Rate"].iloc[-1])  # last row's refresh value
            else:
                refresh_rate = 10  # default fallback

            # ✅ Instant prediction
            if all(col in df.columns for col in features):
                latest = df[features].iloc[-1].values.reshape(1, -1)
                prediction = model.predict(latest)[0]
                st.success(f"Current Predicted Glucose: **{round(float(prediction), 2)} mg/dL**")

                # ✅ Collect past glucose predictions for time-series
                if "Predicted_Glucose" not in df.columns:
                    df["Predicted_Glucose"] = [np.nan] * len(df)

                df.loc[df.index[-1], "Predicted_Glucose"] = prediction

                # ✅ Use ARIMA to forecast next 6 readings (~30–60 mins if refresh_rate=5–10min)
                past_values = df["Predicted_Glucose"].dropna().values
                if len(past_values) > 10:  # need at least 10 data points
                    model_arima = ARIMA(past_values, order=(2,1,2))
                    model_fit = model_arima.fit()
                    forecast = model_fit.forecast(steps=6)  # next 6 future steps

                    st.subheader("📊 Glucose Forecast (Next 30–60 min)")
                    st.line_chart(forecast)

                    # ✅ Alerts
                    if np.any(forecast > 180):
                        st.error("🚨 ALERT: Glucose predicted to go HIGH (>180 mg/dL) in next hour!")
                    elif np.any(forecast < 70):
                        st.warning("⚠️ ALERT: Glucose predicted to go LOW (<70 mg/dL) in next hour!")
                    else:
                        st.success("✅ Glucose forecast within safe range.")

            else:
                st.error(f"❌ Sheet must have columns: {features}")

            st.info(f"⏳ Auto-refreshing every {refresh_rate} seconds (from sheet).")

    except Exception as e:
        st.error(f"⚠️ Could not fetch data: {e}")

    time.sleep(refresh_rate)

