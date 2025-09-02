import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import altair as alt
import joblib
from sklearn.linear_model import LinearRegression

# ===== Config =====
SHEET_URL = "https://docs.google.com/spreadsheets/d/1hoNuXaW_y8QPL3Cb8rhUa3ajGnoeKVEfTRVK8OJ1stI/gviz/tq?tqx=out:csv&sheet=Sheet1"
FEATURES = ["AC_Red", "DC_Red", "AC_IR", "DC_IR", "Heart_Rate", "SpO2", "Perfusion_Index"]
LAGS = 12  # how many past points to use as features for the forecaster (≈ last 12 mins by default)

st.set_page_config(page_title="AI Glucose Monitor", layout="wide")
st.title("🧪 Live AI Glucose Monitor Dashboard (ML Forecasts)")

st.markdown("""
- Pulls **live data** from Google Sheets  
- Predicts **current glucose** using your trained model  
- **Forecasts 30 & 60 minutes ahead with ML** (lag-feature regression on your historical predictions)  
- Alerts on out-of-range values  
""")

# Auto-refresh every 10 seconds
_ = st_autorefresh(interval=10 * 1000, key="data_refresh")

# --- Utilities ---
def detect_timestamp_column(df):
    for c in ["Timestamp", "Time", "Datetime", "DateTime"]:
        if c in df.columns:
            return c
    return None

def minutes_per_row(df, ts_col):
    if ts_col is None:
        return 1.0  # assume 1 row = 1 minute
    try:
        t = pd.to_datetime(df[ts_col], errors="coerce")
        dt = t.diff().dropna().dt.total_seconds() / 60.0
        if len(dt) == 0:
            return 1.0
        # use robust central tendency
        return float(np.nanmedian(dt.clip(lower=0.1)))  # avoid zeros
    except Exception:
        return 1.0

def make_supervised(series: np.ndarray, lags: int, horizon_rows: int):
    """
    Build (X, y) where each X has [t-1, t-2, ..., t-lags] and y is value at t+horizon_rows.
    """
    X, y = [], []
    for i in range(lags, len(series) - horizon_rows):
        X.append(series[i-lags:i][::-1])  # last LAGS, most recent first
        y.append(series[i + horizon_rows])
    return np.asarray(X), np.asarray(y)

def train_forecaster(history: np.ndarray, lags: int, horizon_rows: int):
    X, y = make_supervised(history, lags, horizon_rows)
    if len(X) < 8:  # not enough data to train sensibly
        return None, (len(X), len(y))
    model = LinearRegression()
    model.fit(X, y)
    return model, (len(X), len(y))

def recursive_forecast(history: np.ndarray, lags: int, models_by_h):
    """
    Given trained models keyed by horizon_rows, generate predictions for each horizon
    using the available history and lag window. If a model is None (not enough data),
    fall back to a simple linear fit on recent history.
    """
    preds = {}
    recent = history.copy().astype(float)

    # Fall-back linear slope on last window
    def fallback_predict(h_rows):
        window = min(len(recent), max(lags, 6))
        x = np.arange(window).reshape(-1, 1)
        y = recent[-window:]
        lin = LinearRegression().fit(x, y)
        return float(lin.predict([[window - 1 + h_rows]])[0])

    for h, model in models_by_h.items():
        if (model is None) or (len(recent) < lags):
            preds[h] = fallback_predict(h)
            continue

        # Build feature vector from the latest LAGS values (most recent first)
        x_last = recent[-lags:][::-1].reshape(1, -1)
        preds[h] = float(model.predict(x_last)[0])

    return preds

# --- App logic ---
try:
    df = pd.read_csv(SHEET_URL)

    st.subheader("📊 Recent Data")
    st.dataframe(df.tail(5))

    # Validate features
    if not all(col in df.columns for col in FEATURES):
        st.error(f"❌ Missing required columns in sheet: {FEATURES}")
        st.stop()

    # Load your trained model for "current" prediction
    model_current = joblib.load("glucose_model.pkl")

    # Predict glucose for all rows (historical series)
    X_all = df[FEATURES].values
    df["Predicted_Glucose"] = model_current.predict(X_all).astype(float)

    # Current prediction (latest row)
    current_glucose = float(df["Predicted_Glucose"].iloc[-1])

    # --- Build ML forecaster on historical predictions ---
    ts_col = detect_timestamp_column(df)
    m_per_row = minutes_per_row(df, ts_col)

    # How many rows correspond to 30 / 60 min?
    h30_rows = max(1, int(round(30.0 / m_per_row)))
    h60_rows = max(h30_rows + 1, int(round(60.0 / m_per_row)))  # ensure strictly farther than 30min

    history = df["Predicted_Glucose"].to_numpy()

    model_30, _ = train_forecaster(history, LAGS, h30_rows)
    model_60, _ = train_forecaster(history, LAGS, h60_rows)

    preds = recursive_forecast(history, LAGS, {h30_rows: model_30, h60_rows: model_60})
    glucose_30 = preds[h30_rows]
    glucose_60 = preds[h60_rows]

    # --- Alerts ---
    if glucose_30 > 180 or glucose_60 > 180:
        st.error("🚨 Alert: Glucose predicted to go TOO HIGH (>180 mg/dL)!")
    elif glucose_30 < 70 or glucose_60 < 70:
        st.error("🚨 Alert: Glucose predicted to go TOO LOW (<70 mg/dL)!")
    else:
        st.success("🟢 Glucose levels are within the safe range.")

    # --- Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Glucose", f"{current_glucose:.2f} mg/dL")
    col2.metric("In 30 min (ML)", f"{glucose_30:.2f} mg/dL",
                delta=f"{glucose_30 - current_glucose:.2f}")
    col3.metric("In 60 min (ML)", f"{glucose_60:.2f} mg/dL",
                delta=f"{glucose_60 - current_glucose:.2f}")

    # --- Chart: History + ML Forecast points ---
    st.subheader("📈 Glucose Trends & ML Forecast")

    # Historical
    hist_df = df.reset_index().rename(columns={"index": "Record"})
    hist_plot = hist_df[["Record", "Predicted_Glucose"]].copy()
    hist_plot["Type"] = "History"

    # Forecast points are at future "Record" indices
    last_idx = int(hist_plot["Record"].iloc[-1])
    fc_df = pd.DataFrame({
        "Record": [last_idx + h30_rows, last_idx + h60_rows],
        "Predicted_Glucose": [glucose_30, glucose_60],
        "Type": ["Forecast (30m)", "Forecast (60m)"]
    })

    chart_df = pd.concat([hist_plot, fc_df], ignore_index=True)

    line = alt.Chart(chart_df[chart_df["Type"] == "History"]).mark_line(point=True).encode(
        x="Record:Q",
        y="Predicted_Glucose:Q",
        color=alt.value("#6aa9ff"),
        tooltip=["Record", "Predicted_Glucose"]
    )
    points = alt.Chart(fc_df).mark_point(size=100).encode(
        x="Record:Q",
        y="Predicted_Glucose:Q",
        color="Type:N",
        tooltip=["Type", "Predicted_Glucose"]
    )

    st.altair_chart((line + points).properties(height=420), use_container_width=True)

except Exception as e:
    st.error(f"❌ Error: {e}")
