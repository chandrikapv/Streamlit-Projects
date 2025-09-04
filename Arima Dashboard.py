# streamlit_app.py
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# --- Load ARIMA model ---
with open("arima_model.sav", "rb") as f:
    model = pickle.load(f)

st.title("📊 Sales Forecast Dashboard (ARIMA + Streamlit)")

# --- User inputs ---
horizon = st.slider("Forecast horizon (months ahead)", min_value=1, max_value=24, value=12)

# --- Run forecast ---
forecast = model.get_forecast(steps=horizon)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# --- Prepare dataframe ---
forecast_df = pd.DataFrame({
    "Forecast": forecast_mean,
    "Lower CI": conf_int.iloc[:, 0],
    "Upper CI": conf_int.iloc[:, 1]
})

st.subheader("Forecast Table")
st.dataframe(forecast_df)

# --- Plot ---
fig, ax = plt.subplots()
forecast_mean.plot(ax=ax, label="Forecast", color="blue")
ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="blue", alpha=0.2)
ax.set_title("Sales Forecast with Confidence Interval")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
st.pyplot(fig)

st.markdown("⚡ Adjust the slider to change forecast horizon.")
