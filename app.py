import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load the model
@st.cache_resource
def load_illegal_fishing_model():
    model = load_model("model/InceptionTime_best_model.h5")
    return model

model = load_illegal_fishing_model()

st.title("🌊 Illegal Fishing Detection System")
st.write("Upload sensor time-series data to detect illegal fishing activity using InceptionTime model.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(data.head())

        # Data Preprocessing (adjust this part based on your model's expected input)
        input_data = np.array(data).astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)  # shape: (1, timesteps, features)

        # Predict
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction, axis=1)[0]

        st.success(f"Prediction: {'Illegal Activity Detected' if predicted_class == 1 else 'No Illegal Activity'}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
