import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ✅ Move this to the top!
st.set_page_config(page_title="Cucumber Leaf Disease Classifier", page_icon="🥒")

# Load the trained model
@st.cache_resource
def load_vgg16_model():
    model = load_model("vgg_net16.h5")  # Change filename if needed
    return model

model = load_vgg16_model()

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to VGG16 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Streamlit UI Design
st.set_page_config(page_title="Cucumber Leaf Disease Classifier", page_icon="🥒")

st.title("🥒 Cucumber Leaf Disease Classifier 🍃")
st.write("Upload an image of a cucumber leaf to see if it is **Healthy** or **Diseased**.")

# Upload file button
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_img = preprocess_image(img)

    # Make prediction
    prediction = model.predict(processed_img)
    predicted_class = "Healthy 🍀" if prediction[0][0] > 0.5 else "Diseased ⚠️"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

    # Display results beautifully
    st.subheader("Prediction Result:")
    st.markdown(f"### {predicted_class}")
    st.markdown(f"🧐 **Confidence:** {confidence:.2%}")

    if predicted_class == "Diseased ⚠️":
        st.warning("🚨 This leaf appears to be diseased. Consider taking precautions!")
    else:
        st.success("✅ This leaf looks healthy!")

