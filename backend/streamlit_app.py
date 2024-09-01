import streamlit as st
from PIL import Image
import cv2
import os
import numpy as np
import shutil
from tensorflow.keras.models import load_model

# Load your ML model
model = load_model('D:\\Hackathon_project\\backend\\model.keras')

# Create a temporary directory to save uploaded images
uploaded_folder = "uploaded_images"
if not os.path.exists(uploaded_folder):
    os.makedirs(uploaded_folder)

def process_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(image, (180, 180))
    normalized_image = resized_image / 255.0
    normalized_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension

    result = model.predict(normalized_image)
    predicted_class = np.argmax(result, axis=1)[0]

    return 'defect_detected' if predicted_class == 0 else 'no_defect'

def process_folder(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            image = Image.open(file_path)
            result = process_image(image)
            results.append((filename, result))
    return results

uploaded_files = st.file_uploader("Upload images for defect detection", accept_multiple_files=True)

if st.button("Process Uploaded Folder"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(uploaded_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        with st.spinner('Processing images...'):
            results = process_folder(uploaded_folder)
        
        # Display the results
        for filename, result in results:
            st.write(f"Image: {filename}, Result: {result}")
        
        # Optionally, clean up the uploaded files after processing
        shutil.rmtree(uploaded_folder)
        os.makedirs(uploaded_folder)
    else:
        st.error("Please upload at least one image.")

# Inject custom CSS for an animated gradient background
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(45deg, #3b6978, #204051, #84a9ac, #cae8d5);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp {
        background: rgba(255, 255, 255, 0.8); 
        padding: 20px;
        border-radius: 15px;
        max-width: 800px;
        margin: auto;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)
