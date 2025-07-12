import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your saved model
model = tf.keras.models.load_model("illegal_fishing_model.h5")

# Class labels (edit based on your model)
class_names = ['Class A', 'Class B', 'Class C']

# Title
st.title("🧠 Image Analyzer with ML Model")

# Image uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image (resize based on model input)
    image_resized = image.resize((224, 224))  # change size to match your model input
    img_array = np.array(image_resized) / 255.0  # normalize if needed
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    # Output
    st.subheader("Prediction:")
    st.success(f"Predicted Class: {predicted_class}")
