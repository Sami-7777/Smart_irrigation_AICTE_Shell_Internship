import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Smart Irrigation System", layout="centered")

#  Load the trained model
MODEL_PATH = "Farm_Irrigation_System.pkl"

st.title("🌿 Smart Sprinkler Prediction System")
st.markdown("Enter scaled **sensor values** (0 to 1) below to predict which sprinklers should be ON or OFF based on your trained ML model.")

#  Safe loading with error message
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file `{MODEL_PATH}` not found. Please upload the `.pkl` file.")
    st.stop()

#  Collect sensor inputs (grouped into columns)
sensor_values = []
st.subheader("Sensor Input Panel")
cols = st.columns(4)  # Display 5 sensors per column

for i in range(20):
    col = cols[i % 4]
    val = col.slider(f"Sensor {i}", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    sensor_values.append(val)

#  Predict button
if st.button("🚀 Predict Sprinklers"):
    input_array = np.array(sensor_values).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    on_count = sum(prediction)

    st.success(f"✅ {on_count} sprinkler(s) will be turned ON.")

    st.markdown("---")
    st.markdown("### 🌱 Sprinkler Prediction Status")

    for i, status in enumerate(prediction):
        if status == 1:
            st.markdown(f"🔵 Sprinkler **{i}** (parcel_{i}): **ON**")
        else:
            st.markdown(f"⚪ Sprinkler **{i}** (parcel_{i}): OFF")
