# import pandas as pd
# import numpy as np
# from statsmodels.tsa.arima.model import ARIMA
# import matplotlib.pyplot as plt

# # -------------------------------
# # Step 1: Load / Collect Data
# # -------------------------------
# # Example glucose data every 5 minutes
# # Replace this with live readings from Arduino/ESP32
# glucose_data = [110, 115, 118, 120, 125, 128, 130, 135, 138, 140]

# time_index = pd.date_range(start="2025-09-01 10:00", periods=len(glucose_data), freq="5T")
# series = pd.Series(glucose_data, index=time_index)

# # -------------------------------
# # Step 2: Build & Train Model
# # -------------------------------
# # ARIMA(p,d,q) – tuned for glucose trend forecasting
# model = ARIMA(series, order=(2,1,2))
# model_fit = model.fit()

# # -------------------------------
# # Step 3: Forecast Next 30–60 min
# # -------------------------------
# # If each sample = 5 min, then 6 steps = 30 min, 12 steps = 60 min
# forecast_30min = model_fit.forecast(steps=6)   # 30 min ahead
# forecast_60min = model_fit.forecast(steps=12)  # 60 min ahead

# # -------------------------------
# # Step 4: Alerts
# # -------------------------------
# safe_min = 80    # mg/dL (hypoglycemia threshold)
# safe_max = 180   # mg/dL (hyperglycemia threshold)

# def check_alerts(predicted, horizon):
#     last_val = predicted.iloc[-1]
#     if last_val < safe_min:
#         print(f"⚠ ALERT: Glucose may go LOW ({last_val:.1f} mg/dL) in {horizon} minutes!")
#     elif last_val > safe_max:
#         print(f"⚠ ALERT: Glucose may go HIGH ({last_val:.1f} mg/dL) in {horizon} minutes!")
#     else:
#         print(f"✅ Glucose stable ({last_val:.1f} mg/dL) for next {horizon} minutes.")

# check_alerts(forecast_30min, 30)
# check_alerts(forecast_60min, 60)

# # -------------------------------
# # Step 5: Visualization
# # -------------------------------
# plt.figure(figsize=(10,5))
# plt.plot(series, label="Actual Data")
# plt.plot(forecast_60min, label="Predicted (Next 60 min)", linestyle="--")
# plt.axhline(y=safe_min, color="blue", linestyle="--", label="Low Threshold")
# plt.axhline(y=safe_max, color="red", linestyle="--", label="High Threshold")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Glucose (mg/dL)")
# plt.title("Glucose Prediction (Next 30–60 min)")
# plt.show()

import pandas as pd
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Connect to Google Sheet
# -------------------------------
# Define the scope
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Load credentials (downloaded from Google Cloud)
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

# Open Google Sheet by name
sheet = client.open("GlucoseData").sheet1   # <-- change to your sheet name

# Fetch all rows (assume column A = time, column B = glucose)
data = sheet.get_all_records()

# Convert to DataFrame
df = pd.DataFrame(data)

# Ensure correct data types
df["Time"] = pd.to_datetime(df["Time"])
df.set_index("Time", inplace=True)
series = df["Glucose"]

print("✅ Data loaded from Google Sheets:")
print(series.tail())

# -------------------------------
# Step 2: Train ARIMA Model
# -------------------------------
model = ARIMA(series, order=(2,1,2))
model_fit = model.fit()

# -------------------------------
# Step 3: Forecast Next 30–60 min
# -------------------------------
forecast_30min = model_fit.forecast(steps=6)   # 30 min ahead (6x5min)
forecast_60min = model_fit.forecast(steps=12)  # 60 min ahead (12x5min)

# -------------------------------
# Step 4: Alerts
# -------------------------------
safe_min = 80   # mg/dL
safe_max = 180  # mg/dL

def check_alerts(predicted, horizon):
    last_val = predicted.iloc[-1] if hasattr(predicted, "iloc") else predicted[-1]
    if last_val < safe_min:
        print(f"⚠ ALERT: Glucose may go LOW ({last_val:.1f} mg/dL) in {horizon} minutes!")
    elif last_val > safe_max:
        print(f"⚠ ALERT: Glucose may go HIGH ({last_val:.1f} mg/dL) in {horizon} minutes!")
    else:
        print(f"✅ Glucose stable ({last_val:.1f} mg/dL) for next {horizon} minutes.")

check_alerts(forecast_30min, 30)
check_alerts(forecast_60min, 60)

# -------------------------------
# Step 5: Visualization
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(series, label="Actual Data")
plt.plot(forecast_60min, label="Predicted (Next 60 min)", linestyle="--")
plt.axhline(y=safe_min, color="blue", linestyle="--", label="Low Threshold")
plt.axhline(y=safe_max, color="red", linestyle="--", label="High Threshold")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Glucose (mg/dL)")
plt.title("Glucose Prediction (Next 30–60 min)")
plt.show()

