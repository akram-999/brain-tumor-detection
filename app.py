import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model.h5")

model = load_model()

st.title("🧠 Brain Tumor Detection")
st.write("Upload a brain MRI image and let the model predict whether a tumor is present.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)


    # Resize and preprocess
    resized_img = ImageOps.fit(image, (128, 128), method=Image.Resampling.LANCZOS)
    img_array = np.asarray(resized_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict with spinner
    with st.spinner("🔍 Analyzing the image..."):
        prediction = model.predict(img_array)[0][0]
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        result = "🧠 Tumor Detected" if prediction > 0.5 else "✅ No Tumor Detected"
        bar_color = "red" if prediction > 0.5 else "green"

    # Display prediction
    st.markdown(f"### Prediction: **{result}**")
    st.progress(int(confidence * 100))
    st.markdown(f"**Confidence: {confidence:.2f}**")
