import streamlit as st

# Set the page configuration at the very beginning
st.set_page_config(layout="wide", page_title="Cucumber Leaf Disease Classification")

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

st.title("Cucumber Leaf Disease Classification Dashboard")

# Creating two columns for a split-screen layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload an Image")
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

        if confidence >= 50:
            final_prediction = predicted_class + ": Powdery Mildew Disease"
            st.write(f"**Prediction:** {final_prediction}")
            st.write(f"**Confidence:** {confidence:.2f}%")
        else:
            final_prediction = "Healthy"
            st.write(f"**Prediction:** {final_prediction}")
            st.write(f"**Confidence:** {100 - confidence:.2f}%")

with col2:
    st.subheader("Cucumber Powdery Mildew")
    st.markdown(
        """
        **Etiology:**
        Cucumber powdery mildew is primarily caused by two fungal pathogens:
        - **Podosphaera xanthii** (formerly known as *Sphaerotheca fuliginea*) – Most common causal agent.
        - **Golovinomyces cichoracearum** (formerly known as *Erysiphe cichoracearum*) – Less common but still found in some regions.
        
        **Fungicide Recommendations Based on Disease Severity:**
        - **Mild Infection (Initial Spots Observed):**
            - *Azoxystrobin 23% SC* – 1.0 ml/L *(Waiting period: 5 days)*
            - *Metrafenone 50% SC* – 0.5 ml/L *(Waiting period: 35 days)*
        - **Moderate to Severe Infection (Spreading to Multiple Leaves):**
            - *Azoxystrobin 4.8% w/w + Chlorothalonil 40% w/w SC* – 1.8 ml/L *(Waiting period: 3 days)*
            - *Fluxapyroxad 250 g/L + Pyraclostrobin 250 g/L SC* – 0.4-0.5 ml/L *(Waiting period: 10 days)*
        """
    )

st.markdown("---")
st.write("Developed by Anurag Dhole")
st.write("             Dr. Jadesha Mandya")
st.write("             Dr. Deepak D.")
