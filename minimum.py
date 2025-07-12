import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("your_model.h5")
    return model

model = load_model()

# Replace with your actual class names
class_names = ['Class A', 'Class B', 'Class C']

# App title
st.title("🧠 Image Classifier")

# Image upload section
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image_resized = image.resize((224, 224))  # Change size if your model uses a different one
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Show result
    st.subheader("Prediction:")
    st.success(f"Predicted class: {predicted_class}")
