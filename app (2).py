import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="AI Battery Intelligence",
    layout="wide"
)

st.title("AI Battery Intelligence")
st.subheader("Battery Health & Degradation Prediction")

# Load dataset and trained model
data = pd.read_csv("nasa_battery_data.csv")
model = joblib.load("battery_capacity_model.pkl")

battery_names = sorted(data["battery"].unique())
selected = st.selectbox("Select Battery", battery_names)

df = data[data["battery"] == selected].sort_values("cycle")

initial_capacity = df["capacity_Ah"].iloc[0]
latest_capacity = df["capacity_Ah"].iloc[-1]
soh = latest_capacity / initial_capacity * 100

col1, col2, col3 = st.columns(3)
col1.metric("Battery", selected)
col2.metric("Latest Capacity", f"{latest_capacity:.3f} Ah")
col3.metric("Estimated SOH", f"{soh:.1f}%")

st.subheader("Battery Capacity Degradation")

fig, ax = plt.subplots()
ax.plot(df["cycle"], df["capacity_Ah"], marker="o")
ax.set_xlabel("Cycle")
ax.set_ylabel("Capacity (Ah)")
ax.grid(True)
st.pyplot(fig)
plt.close(fig)

st.subheader("ML-Based Capacity Prediction")

max_cycle = int(data["cycle"].max())
future_cycle = st.number_input(
    "Enter cycle number to predict",
    min_value=0,
    max_value=max_cycle + 500,
    value=max_cycle + 10
)

predicted_capacity = float(
    model.predict(pd.DataFrame({"cycle": [future_cycle]}))[0]
)

st.metric(
    "Predicted Capacity",
    f"{predicted_capacity:.3f} Ah"
)

st.caption(
    "This is an experimental ML estimate based on historical NASA battery data, "
    "not a live measurement of a physical battery."
)

st.download_button(
    "Download Battery Data",
    df.to_csv(index=False),
    file_name=f"{selected}_battery_data.csv",
    mime="text/csv"
)
