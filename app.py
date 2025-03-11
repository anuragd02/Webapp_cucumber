import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Google Drive file ID
MODEL_FILE_ID = "1TdP0VR0ZpHrLSzi34TIsyDbOUBS74u80"
MODEL_PATH = "vgg_net16.h5"

# Function to download the model from Google Drive
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):  # Download only if the model is not already present
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    # Load the model
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load the model
model = download_and_load_model()

# Define class labels
CLASS_NAMES = ["Unhealthy", "Healthy"]

st.title("Cucumber Leaf Disease Classification 🍃")
st.write("Upload an image of a cucumber leaf to classify whether it's **Healthy** or **Unhealthy**.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = image.resize((224, 224))  # Ensure this matches the model's expected input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    confidence = np.max(predictions) * 100
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    # Display results
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
