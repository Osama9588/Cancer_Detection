import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Load the trained model
model_path = "updated_model.keras"
loaded_model = tf.keras.models.load_model(model_path)

# Streamlit app title
st.title("Breast Cancer Prediction")

# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type=["png","jpeg","jpg"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    
    # Preprocess the image (convert to RGB, resize, normalize)
    image = image.convert('RGB')
    image = image.resize((224, 224))
    input_data = np.expand_dims(image, axis=0)  # Add batch dimension
    input_data = np.array(input_data) / 255.0  # Normalize

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make a prediction
    pred = loaded_model.predict(input_data)

    # Display the result
    if pred >= 0.5:
        st.write("Prediction: **Cancer**")
    else:
        st.write("Prediction: **No Cancer**")
