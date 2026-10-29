st.subheader("🔋 Battery Health Status")

battery_soh = float(soh)

if battery_soh > 90:
    st.success("🟢 Healthy Battery — Battery health is good.")
elif battery_soh > 80:
    st.warning("🟡 Moderate Battery — Battery health is declining.")
else:
    st.error("🔴 Critical Battery — Battery health is below 80%.")

st.progress(max(0.0, min(1.0, battery_soh / 100)))
st.caption(f"Current Battery Health: {battery_soh:.1f}%")

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
