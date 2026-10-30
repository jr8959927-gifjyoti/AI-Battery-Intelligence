import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AI Battery Intelligence",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 AI Battery Intelligence")
st.subheader("Battery Health, Capacity & RUL Prediction")

st.write(
    "An experimental machine learning application for "
    "battery health monitoring and degradation prediction."
)

# ---------------- LOAD DATA AND MODELS ----------------
try:
    data = pd.read_csv("nasa_battery_data.csv")
    capacity_model = joblib.load("battery_capacity_model.pkl")
    rul_model = joblib.load("battery_rul_model.pkl")

except FileNotFoundError as e:
    st.error(
        f"File not found: {e.filename}. Please check that "
        "the dataset and both model files are in your repository."
    )
    st.stop()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

# ---------------- VALIDATE DATA ----------------
required_columns = ["battery", "cycle", "capacity_Ah"]
missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()

data["cycle"] = pd.to_numeric(data["cycle"], errors="coerce")
data["capacity_Ah"] = pd.to_numeric(
    data["capacity_Ah"], errors="coerce"
)

data = data.dropna(
    subset=["battery", "cycle", "capacity_Ah"]
).copy()

if data.empty:
    st.error("No valid battery records were found.")
    st.stop()

if "SOH_percent" in data.columns:
    data["SOH_percent"] = pd.to_numeric(
        data["SOH_percent"], errors="coerce"
    )

# ---------------- SELECT BATTERY ----------------
st.sidebar.header("Battery Selection")

battery_names = sorted(data["battery"].unique())

selected_battery = st.sidebar.selectbox(
    "Choose a battery",
    battery_names
)

df = (
    data[data["battery"] == selected_battery]
    .sort_values("cycle")
    .copy()
)

if df.empty:
    st.error("No data found for the selected battery.")
    st.stop()

latest = df.iloc[-1]

latest_capacity = float(latest["capacity_Ah"])
current_cycle = int(latest["cycle"])

# ---------------- CALCULATE SOH ----------------
initial_capacity = float(df["capacity_Ah"].iloc[0])

if initial_capacity <= 0:
    st.error("Initial battery capacity is invalid.")
    st.stop()

if (
    "SOH_percent" in df.columns
    and pd.notna(latest["SOH_percent"])
):
    soh = float(latest["SOH_percent"])
else:
    soh = (latest_capacity / initial_capacity) * 100

# ---------------- BATTERY SUMMARY ----------------
st.header("📊 Battery Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Selected Battery",
    str(selected_battery)
)

col2.metric(
    "Latest Capacity",
    f"{latest_capacity:.3f} Ah"
)

col3.metric(
    "State of Health (SOH)",
    f"{soh:.1f}%"
)

# ---------------- HEALTH STATUS ----------------
st.header("🔋 Battery Health Status")

if soh > 90:
    st.success("🟢 Healthy Battery — Health is above 90%.")
elif soh > 80:
    st.warning("🟡 Moderate Battery — Health is above 80% and up to 90%.")
else:
    st.error("🔴 Critical Battery — Health is at or below 80%.")

st.progress(max(0.0, min(1.0, soh / 100)))
st.caption(f"Current recorded battery health: {soh:.1f}%")

# ---------------- CAPACITY DEGRADATION GRAPH ----------------
st.header("📉 Battery Capacity Degradation")

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    df["cycle"],
    df["capacity_Ah"],
    marker=".",
    linewidth=1.5,
    label="Recorded Capacity"
)

ax.set_xlabel("Cycle Number")
ax.set_ylabel("Capacity (Ah)")
ax.set_title(f"Capacity Degradation — {selected_battery}")
ax.grid(True)
ax.legend()
st.pyplot(fig)
plt.close(fig)

# ---------------- CAPACITY PREDICTION ----------------
st.header("🤖 ML-Based Capacity Prediction")

max_cycle = int(data["cycle"].max())

future_cycle = st.number_input(
    "Enter cycle number for capacity prediction",
    min_value=0,
    max_value=max_cycle + 500,
    value=max_cycle + 10,
    step=1
)

try:
    capacity_input = pd.DataFrame({
        "cycle": [future_cycle]
    })

    predicted_capacity = float(
        capacity_model.predict(capacity_input)[0]
    )

    st.metric(
        "Predicted Capacity",
        f"{predicted_capacity:.3f} Ah"
    )

    st.caption(
        "Future predictions are estimates. Accuracy may decrease "
        "when predicting beyond the training data."
    )

except Exception as e:
    st.error(f"Capacity prediction failed: {e}")

# ---------------- RUL PREDICTION ----------------
st.header("⏳ Remaining Useful Life (RUL) Prediction")

st.write(f"Current recorded cycle: **{current_cycle}**")

if "SOH_percent" not in data.columns:
    st.warning(
        "SOH_percent is missing from the dataset. "
        "The RUL model expects this feature."
    )
else:
    try:
        rul_input = pd.DataFrame([{
            "cycle": current_cycle,
            "SOH_percent": soh,
            "capacity_Ah": latest_capacity
        }])

        predicted_rul = float(
            rul_model.predict(rul_input)[0]
        )

        predicted_rul = max(0.0, predicted_rul)

        col4, col5 = st.columns(2)

        col4.metric(
            "Estimated Remaining Cycles",
            f"{predicted_rul:.0f}"
        )

        col5.metric(
            "Estimated EOL Cycle",
            f"{current_cycle + predicted_rul:.0f}"
        )

        st.info(
            "RUL is an experimental model estimate, not a "
            "guaranteed battery lifetime."
        )

    except Exception as e:
        st.error(f"RUL prediction failed: {e}")

# ---------------- ACTUAL VS PREDICTED GRAPH ----------------
st.header("📈 Actual vs Predicted Capacity")

st.write(
    "Compare recorded capacity with model predictions "
    "for the selected battery."
)

try:
    comparison_df = df[["cycle", "capacity_Ah"]].dropna().copy()

    if len(comparison_df) < 2:
        st.warning(
            "Not enough records to draw the comparison graph."
        )
    else:
        comparison_df["Predicted Capacity"] = capacity_model.predict(
            comparison_df[["cycle"]]
        )

        fig2, ax2 = plt.subplots(figsize=(10, 4))

        ax2.plot(
            comparison_df["cycle"],
            comparison_df["capacity_Ah"],
            label="Actual Capacity",
            marker=".",
            linewidth=1.5
        )

        ax2.plot(
            comparison_df["cycle"],
            comparison_df["Predicted Capacity"],
            label="Predicted Capacity",
            linestyle="--",
            linewidth=1.5
        )

        ax2.set_xlabel("Cycle Number")
        ax2.set_ylabel("Capacity (Ah)")
        ax2.set_title("Actual vs Predicted Capacity")
        ax2.grid(True)
        ax2.legend()

        st.pyplot(fig2)
        plt.close(fig2)

except Exception as e:
    st.error(f"Comparison graph failed: {e}")

# ---------------- MODEL PERFORMANCE METRICS ----------------
st.header("📋 Capacity Model Performance")

st.write(
    "These metrics compare predictions with recorded values "
    "for the selected battery. They are descriptive only and "
    "are not an independent test of model performance."
)

try:
    metric_df = df[["cycle", "capacity_Ah"]].dropna().copy()

    if len(metric_df) >= 2:
        actual = metric_df["capacity_Ah"].to_numpy()

        predicted = capacity_model.predict(
            metric_df[["cycle"]]
        )

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        if len(np.unique(actual)) > 1:
            r2 = r2_score(actual, predicted)
        else:
            r2 = None

        m1, m2, m3 = st.columns(3)

        m1.metric("MAE", f"{mae:.4f} Ah")
        m2.metric("RMSE", f"{rmse:.4f} Ah")

        if r2 is not None:
            m3.metric("R² Score", f"{r2:.4f}")
        else:
            m3.metric("R² Score", "Not available")

    else:
        st.info("Not enough records to calculate these metrics.")

except Exception as e:
    st.warning(f"Could not calculate metrics: {e}")

# ---------------- DOWNLOAD DATA ----------------
st.header("📥 Download Battery Data")

csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Selected Battery CSV",
    data=csv_data,
    file_name=f"{selected_battery}_battery_data.csv",
    mime="text/csv"
)

# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "AI Battery Intelligence | Experimental software prototype. "
    "This application uses historical dataset records and does "
    "not monitor a physical battery in real time."
)
# -----------------------------------------
# BATTERY CHARGING & DISCHARGING TIME ESTIMATOR
# -----------------------------------------

import streamlit as st

st.header("🔋 Battery Charging & Discharging Time Estimator")

st.write(
    "Estimate the approximate charging and discharging time "
    "of a Lithium-Ion battery."
)

# Battery capacity
capacity = st.number_input(
    "Battery Capacity (Ah)",
    min_value=0.1,
    value=2.0,
    step=0.1
)

# Current battery charge level
soc = st.slider(
    "Current Battery Charge Level (%)",
    min_value=0,
    max_value=100,
    value=30
)

# Target charging level
target_soc = st.slider(
    "Target Charging Level (%)",
    min_value=1,
    max_value=100,
    value=100
)

# Charging current
charging_current = st.number_input(
    "Charging Current (A)",
    min_value=0.1,
    value=1.0,
    step=0.1
)

# Discharging load current
load_current = st.number_input(
    "Discharging Load Current (A)",
    min_value=0.1,
    value=1.0,
    step=0.1
)

# Charging efficiency
efficiency = 0.90

if st.button("Calculate Battery Time"):

    if target_soc <= soc:
        st.warning("Target charging level must be higher than current charge level.")

    else:
        # Charging time
        charging_time = (
            capacity * (target_soc - soc) / 100
        ) / (charging_current * efficiency)

        # Discharging time
        discharging_time = (
            capacity * soc / 100
        ) / load_current

        st.subheader("Battery Time Estimation")

        st.success(
            f"Estimated Charging Time: {charging_time:.2f} hours"
        )

        st.info(
            f"Estimated Discharging Time: {discharging_time:.2f} hours"
        )

        st.caption(
            "Note: These are simplified estimates. Actual time depends "
            "on battery condition, voltage, temperature, charging profile "
            "and device load."
        )
  

