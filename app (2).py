import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="AI Battery Intelligence",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 AI Battery Intelligence")
st.subheader("Battery Health, Capacity & RUL Prediction")

# Load dataset and trained models
data = pd.read_csv("nasa_battery_data.csv")
capacity_model = joblib.load("battery_capacity_model.pkl")
rul_model = joblib.load("battery_rul_model.pkl")

# Battery selection
battery_names = sorted(data["battery"].unique())
selected = st.selectbox("Select Battery", battery_names)

df = data[data["battery"] == selected].sort_values("cycle")
latest = df.iloc[-1]

initial_capacity = float(df["capacity_Ah"].iloc[0])
latest_capacity = float(latest["capacity_Ah"])
current_cycle = int(latest["cycle"])

# Use recorded SOH when available
if "SOH_percent" in df.columns:
    soh = float(latest["SOH_percent"])
else:
    soh = (latest_capacity / initial_capacity) * 100

# Capacity and health metrics
col1, col2, col3 = st.columns(3)
col1.metric("Selected Battery", selected)
col2.metric("Latest Capacity", f"{latest_capacity:.3f} Ah")
col3.metric("State of Health (SOH)", f"{soh:.1f}%")

# Capacity degradation chart
st.subheader("Battery Capacity Degradation")

fig, ax = plt.subplots()
ax.plot(df["cycle"], df["capacity_Ah"], marker="o")
ax.set_xlabel("Cycle")
ax.set_ylabel("Capacity (Ah)")
ax.grid(True)
st.pyplot(fig)
plt.close(fig)

# Capacity prediction
st.subheader("ML-Based Capacity Prediction")

max_cycle = int(data["cycle"].max())

future_cycle = st.number_input(
    "Enter cycle number for capacity prediction",
    min_value=0,
    max_value=max_cycle + 500,
    value=max_cycle + 10
)

predicted_capacity = float(
    capacity_model.predict(
        pd.DataFrame({"cycle": [future_cycle]})
    )[0]
)

st.metric("Predicted Capacity", f"{predicted_capacity:.3f} Ah")

# RUL prediction
st.subheader("Remaining Useful Life (RUL) Prediction")

st.write(f"Current recorded cycle: **{current_cycle}**")

rul_input = pd.DataFrame([{
    "cycle": current_cycle,
    "SOH_percent": soh,
    "capacity_Ah": latest_capacity
}])

predicted_rul = float(rul_model.predict(rul_input)[0])
predicted_rul = max(0, predicted_rul)

col4, col5 = st.columns(2)
col4.metric("Estimated Remaining Cycles", f"{predicted_rul:.0f}")
col5.metric(
    "Estimated EOL Cycle",
    f"{current_cycle + predicted_rul:.0f}"
)

if soh <= 80:
    st.warning("The selected battery has reached or passed the chosen 80% SOH threshold.")
else:
    st.info(
        "RUL is a model-based estimate using historical battery data, "
        "not a guaranteed battery lifetime."
    )

st.caption(
    "Experimental software prototype. Predictions depend on dataset "
    "quality and model validation; this is not live battery monitoring."
)

# Download selected battery data
st.download_button(
    "Download Battery Data",
    df.to_csv(index=False),
    file_name=f"{selected}_battery_data.csv",
    mime="text/csv"
)
