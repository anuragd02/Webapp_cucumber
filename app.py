import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("vgg_net16.h5")  # Updated model name
    return model

model = load_model()

# Define class labels for cucumber classification
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

    # Classification logic
    if confidence >= 50:
        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")
    else:
        other_confidence = 100 - confidence
        st.write(f"**Prediction:** Healthy")
        st.write(f"**Confidence:** {other_confidence:.2f}%")
