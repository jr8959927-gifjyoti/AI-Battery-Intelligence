import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AI Battery Intelligence",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 AI Battery Intelligence")
st.subheader("Battery Health, Capacity & RUL Prediction")

# ---------------- LOAD FILES ----------------
try:
    data = pd.read_csv("nasa_battery_data.csv")

    capacity_model = joblib.load("battery_capacity_model.pkl")
    rul_model = joblib.load("battery_rul_model.pkl")

except FileNotFoundError as e:
    st.error(
        f"Required file not found: {e.filename}. "
        "Please check that the CSV and both model files "
        "are present in your GitHub repository."
    )
    st.stop()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

# ---------------- CHECK DATA ----------------
required_columns = ["battery", "cycle", "capacity_Ah"]

missing_columns = [
    col for col in required_columns if col not in data.columns
]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()

if data.empty:
    st.error("The battery dataset is empty.")
    st.stop()

data["cycle"] = pd.to_numeric(data["cycle"], errors="coerce")
data["capacity_Ah"] = pd.to_numeric(
    data["capacity_Ah"], errors="coerce"
)
data = data.dropna(subset=["battery", "cycle", "capacity_Ah"])

if data.empty:
    st.error("No valid battery records are available.")
    st.stop()

# ---------------- BATTERY SELECTION ----------------
battery_names = sorted(data["battery"].unique())

selected_battery = st.selectbox(
    "Select Battery",
    battery_names
)

df = (
    data[data["battery"] == selected_battery]
    .sort_values("cycle")
    .copy()
)

if df.empty:
    st.error("No records found for the selected battery.")
    st.stop()

latest = df.iloc[-1]

latest_capacity = float(latest["capacity_Ah"])
current_cycle = int(latest["cycle"])

# ---------------- CALCULATE SOH ----------------
if (
    "SOH_percent" in df.columns
    and pd.notna(latest.get("SOH_percent"))
):
    soh = float(latest["SOH_percent"])
else:
    initial_capacity = float(df["capacity_Ah"].iloc[0])

    if initial_capacity <= 0:
        st.error("Cannot calculate SOH because initial capacity is invalid.")
        st.stop()

    soh = (latest_capacity / initial_capacity) * 100

# ---------------- BATTERY SUMMARY ----------------
st.subheader("📊 Battery Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Selected Battery", str(selected_battery))
col2.metric("Latest Capacity", f"{latest_capacity:.3f} Ah")
col3.metric("State of Health (SOH)", f"{soh:.1f}%")

# ---------------- HEALTH STATUS ALERT ----------------
st.subheader("🔋 Battery Health Status")

if soh > 90:
    st.success("🟢 Healthy Battery — Health is above 90%.")
elif soh > 80:
    st.warning("🟡 Moderate Battery — Health is between 80% and 90%.")
else:
    st.error("🔴 Critical Battery — Health is at or below 80%.")

st.progress(max(0.0, min(1.0, soh / 100)))
st.caption(f"Current Battery Health: {soh:.1f}%")

# ---------------- CAPACITY DEGRADATION GRAPH ----------------
st.subheader("📉 Battery Capacity Degradation")

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
ax.set_title(f"Capacity Degradation - {selected_battery}")
ax.grid(True)
ax.legend()

st.pyplot(fig)
plt.close(fig)

# ---------------- CAPACITY PREDICTION ----------------
st.subheader("🤖 ML-Based Capacity Prediction")

max_cycle = int(data["cycle"].max())

future_cycle = st.number_input(
    "Enter cycle number for capacity prediction",
    min_value=0,
    max_value=max_cycle + 500,
    value=max_cycle + 10,
    step=1
)

try:
    capacity_input = pd.DataFrame({"cycle": [future_cycle]})
    predicted_capacity = float(
        capacity_model.predict(capacity_input)[0]
    )

    st.metric(
        "Predicted Capacity",
        f"{predicted_capacity:.3f} Ah"
    )

    if predicted_capacity < 0:
        st.warning(
            "The model predicted a negative capacity. "
            "This may indicate unreliable extrapolation."
        )

except Exception as e:
    st.error(f"Capacity prediction failed: {e}")

# ---------------- RUL PREDICTION ----------------
st.subheader("⏳ Remaining Useful Life (RUL) Prediction")

st.write(f"Current recorded cycle: **{current_cycle}**")

if "SOH_percent" not in data.columns:
    st.warning(
        "RUL model expects SOH_percent. "
        "Please check your dataset and model features."
    )
else:
    try:
        rul_input = pd.DataFrame([{
            "cycle": current_cycle,
            "SOH_percent": soh,
            "capacity_Ah": latest_capacity
        }])

        predicted_rul = float(rul_model.predict(rul_input)[0])
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
            "RUL is an experimental model-based estimate, "
            "not a guaranteed battery lifetime."
        )

    except Exception as e:
        st.error(f"RUL prediction failed: {e}")
        st.caption(
            "Check that battery_rul_model.pkl was trained using "
            "cycle, SOH_percent and capacity_Ah in that order."
        )

# ---------------- DOWNLOAD DATA ----------------
st.subheader("📥 Download Battery Data")

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
    "Predictions depend on dataset quality and model validation. "
    "This application does not monitor a physical battery in real time."
)
