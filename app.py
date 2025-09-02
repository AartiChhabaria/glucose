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

# 

# import streamlit as st
# import pandas as pd
# import numpy as np
# import time

# # Simulated prediction model (replace with ML model later)
# def predict_future_glucose(current_glucose, minutes):
#     # Simple linear trend (placeholder)
#     # Example: glucose changes by ±0.5 per 10 min
#     trend_per_min = 0.05  
#     return current_glucose + trend_per_min * minutes

# # Google Sheets URL (replace with your sheet’s published CSV link)
# SHEET_URL = "https://docs.google.com/spreadsheets/d/1hoNuXaW_y8QPL3Cb8rhUa3ajGnoeKVEfTRVK8OJ1stI/gviz/tq?tqx=out:csv&sheet=Sheet1"

# st.set_page_config(page_title="AI Glucose Monitor", layout="wide")

# st.title("🧪 Live AI Glucose Monitor (Google Sheets Input + Future Prediction)")

# st.markdown("""
# This app fetches the latest data from your Google Sheet and:
# - Shows real-time glucose predictions  
# - Forecasts the next 30–60 minutes  
# - Alerts 🚨 if levels are predicted to be too high (>180 mg/dL) or too low (<70 mg/dL)  
# """)

# # Auto-refresh every 10 sec
# REFRESH_SEC = 10
# placeholder = st.empty()

# while True:
#     try:
#         # Load sheet data
#         df = pd.read_csv(SHEET_URL)
        
#         with placeholder.container():
#             st.subheader("📊 Latest data from Google Sheets:")
#             st.dataframe(df.tail(3))  # Show last 3 rows
            
#             # Get latest glucose value
#             current_glucose = np.random.uniform(90, 110)  # <--- replace with ML model result
#             st.success(f"✅ Current Predicted Glucose: **{current_glucose:.2f} mg/dL**")
            
#             # Predictions for 30 and 60 min
#             glucose_30 = predict_future_glucose(current_glucose, 30)
#             glucose_60 = predict_future_glucose(current_glucose, 60)
            
#             st.info(f"🔮 Expected Glucose after 30 min: **{glucose_30:.2f} mg/dL**")
#             st.info(f"🔮 Expected Glucose after 60 min: **{glucose_60:.2f} mg/dL**")
            
#             # Alerts
#             if glucose_30 > 180 or glucose_60 > 180:
#                 st.error("🚨 Alert: Glucose predicted to go TOO HIGH (>180 mg/dL)!")
#             elif glucose_30 < 70 or glucose_60 < 70:
#                 st.error("🚨 Alert: Glucose predicted to go TOO LOW (<70 mg/dL)!")
#             else:
#                 st.success("🟢 Glucose predictions are within the safe range.")
            
#             st.markdown(f"⏳ Auto-refreshing every {REFRESH_SEC} seconds (from sheet).")
        
#     except Exception as e:
#         st.error(f"❌ Error loading sheet: {e}")
    
#     time.sleep(REFRESH_SEC)
# import streamlit as st
# from streamlit_autorefresh import st_autorefresh
# import pandas as pd
# import numpy as np
# import altair as alt

# # Auto-refresh every 10 seconds
# _ = st_autorefresh(interval=10 * 1000, key="data_refresh")

# st.set_page_config(page_title="Glucose Dashboard", layout="wide")

# st.title("Live AI Glucose Monitor Dashboard")

# # ... fetch your data from Google Sheets or wherever
# df = pd.DataFrame()  # Replace with actual data fetch

# current_glucose = np.random.uniform(90, 110)
# glucose_30 = current_glucose + 0.05 * 30
# glucose_60 = current_glucose + 0.05 * 60

# # Alerts
# if glucose_30 > 180 or glucose_60 > 180:
#     st.error("🚨 Predicted glucose too high!")
# elif glucose_30 < 70 or glucose_60 < 70:
#     st.error("🚨 Predicted glucose too low!")
# else:
#     st.success("Glucose levels are within the safe range.")

# # Metrics
# col1, col2, col3 = st.columns(3)
# col1.metric("Current Glucose", f"{current_glucose:.2f} mg/dL")
# col2.metric("In 30 min", f"{glucose_30:.2f} mg/dL", delta=f"{glucose_30 - current_glucose:.2f}")
# col3.metric("In 60 min", f"{glucose_60:.2f} mg/dL", delta=f"{glucose_60 - current_glucose:.2f}")

# # Trend chart
# chart_df = pd.DataFrame({
#     "Time": ["Now", "30 min", "60 min"],
#     "Glucose": [current_glucose, glucose_30, glucose_60]
# })
# chart = alt.Chart(chart_df).mark_line(point=True).encode(
#     x="Time",
#     y="Glucose",
#     tooltip=["Time", "Glucose"]
# ).properties(height=400)

# st.altair_chart(chart, use_container_width=True)

# # Show latest table
# st.subheader("Recent Data")
# st.dataframe(df.tail(5))


# import streamlit as st
# from streamlit_autorefresh import st_autorefresh
# import pandas as pd
# import numpy as np
# import altair as alt

# # Google Sheets link
# SHEET_URL = "https://docs.google.com/spreadsheets/d/1hoNuXaW_y8QPL3Cb8rhUa3ajGnoeKVEfTRVK8OJ1stI/gviz/tq?tqx=out:csv&sheet=Sheet1"

# # Auto-refresh every 10s
# _ = st_autorefresh(interval=10 * 1000, key="data_refresh")

# st.title("🧪 Live AI Glucose Monitor Dashboard")

# try:
#     df = pd.read_csv(SHEET_URL)   # <-- fetch actual data
#     st.subheader("📊 Recent Data")
#     st.dataframe(df.tail(5))      # <-- now this won’t be empty

#     # --- Predictions ---
#     current_glucose = np.random.uniform(90, 110)  # replace with ML model
#     glucose_30 = current_glucose + 0.05 * 30
#     glucose_60 = current_glucose + 0.05 * 60

#     # Alerts
#     if glucose_30 > 180 or glucose_60 > 180:
#         st.error("🚨 Predicted glucose too high!")
#     elif glucose_30 < 70 or glucose_60 < 70:
#         st.error("🚨 Predicted glucose too low!")
#     else:
#         st.success("🟢 Glucose levels are within the safe range.")

#     # Metrics
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Current Glucose", f"{current_glucose:.2f} mg/dL")
#     col2.metric("In 30 min", f"{glucose_30:.2f} mg/dL", delta=f"{glucose_30 - current_glucose:.2f}")
#     col3.metric("In 60 min", f"{glucose_60:.2f} mg/dL", delta=f"{glucose_60 - current_glucose:.2f}")

#     # Chart
#     chart_df = pd.DataFrame({
#         "Time": ["Now", "30 min", "60 min"],
#         "Glucose": [current_glucose, glucose_30, glucose_60]
#     })
#     chart = alt.Chart(chart_df).mark_line(point=True).encode(
#         x="Time", y="Glucose", tooltip=["Time", "Glucose"]
#     )
#     st.altair_chart(chart, use_container_width=True)

# except Exception as e:
#     st.error(f"❌ Error loading data: {e}")
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import altair as alt
import joblib

# --- Load trained model ---
model = joblib.load("glucose_model.pkl")

# --- Google Sheets link (CSV export) ---
SHEET_URL = "https://docs.google.com/spreadsheets/d/1hoNuXaW_y8QPL3Cb8rhUa3ajGnoeKVEfTRVK8OJ1stI/gviz/tq?tqx=out:csv&sheet=Sheet1"

# --- Columns required by model ---
FEATURES = ["AC_Red", "DC_Red", "AC_IR", "DC_IR", "Heart_Rate", "SpO2", "Perfusion_Index"]

# --- Page setup ---
st.set_page_config(page_title="AI Glucose Monitor", layout="wide")
st.title("🧪 Live AI Glucose Monitor Dashboard")

st.markdown("""
This dashboard:
- Pulls **live data** from Google Sheets  
- Predicts **current glucose levels** using your trained ML model  
- Forecasts the next 30–60 minutes  
- Shows **historical glucose trends** alongside predictions  
- Sends 🚨 alerts if abnormal levels are detected  
""")

# --- Auto-refresh every 10 seconds ---
_ = st_autorefresh(interval=10 * 1000, key="data_refresh")

try:
    # --- Load latest data from Google Sheets ---
    df = pd.read_csv(SHEET_URL)

    # Show latest 5 rows
    st.subheader("📊 Recent Data")
    st.dataframe(df.tail(5))

    # --- Check if required features are available ---
    if all(col in df.columns for col in FEATURES):
        # Predict glucose for all past rows (historical trend)
        X = df[FEATURES].values
        df["Predicted_Glucose"] = model.predict(X)

        # Current (latest) glucose
        current_glucose = float(df["Predicted_Glucose"].iloc[-1])

        # Forecast future values (simple placeholder model)
        glucose_30 = current_glucose + 0.05 * 30
        glucose_60 = current_glucose + 0.05 * 60

        # 🚨 Alerts
        if glucose_30 > 180 or glucose_60 > 180:
            st.error("🚨 Alert: Glucose predicted to go TOO HIGH (>180 mg/dL)!")
        elif glucose_30 < 70 or glucose_60 < 70:
            st.error("🚨 Alert: Glucose predicted to go TOO LOW (<70 mg/dL)!")
        else:
            st.success("🟢 Glucose levels are within the safe range.")

        # --- Dashboard Metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Glucose", f"{current_glucose:.2f} mg/dL")
        col2.metric("In 30 min", f"{glucose_30:.2f} mg/dL",
                    delta=f"{glucose_30 - current_glucose:.2f}")
        col3.metric("In 60 min", f"{glucose_60:.2f} mg/dL",
                    delta=f"{glucose_60 - current_glucose:.2f}")

        # --- Historical + Forecast Chart ---
        st.subheader("📈 Glucose Trends & Forecast")

        # Historical chart data
        hist_df = df.reset_index().rename(columns={"index": "Record"})
        hist_df = hist_df[["Record", "Predicted_Glucose"]]

        # Forecast chart data
        forecast_df = pd.DataFrame({
            "Record": [len(hist_df), len(hist_df) + 30, len(hist_df) + 60],
            "Predicted_Glucose": [current_glucose, glucose_30, glucose_60]
        })

        # Combine datasets
        hist_df["Type"] = "History"
        forecast_df["Type"] = "Forecast"
        chart_df = pd.concat([hist_df, forecast_df])

        chart = alt.Chart(chart_df).mark_line(point=True).encode(
            x="Record",
            y="Predicted_Glucose",
            color="Type",
            tooltip=["Record", "Predicted_Glucose", "Type"]
        ).properties(height=400)

        st.altair_chart(chart, use_container_width=True)

    else:
        st.error(f"❌ Missing required columns in sheet: {FEATURES}")

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
