import streamlit as st
import tensorflow as tf
import os
import numpy as np
st.title("3D Printing Defect Detection")
st.write("Upload an image folder for classification")
testing = 'D:\\Hackathon_project\\image'
batch_size = 32
img_height = 180
img_width = 180
AUTOTUNE = tf.data.AUTOTUNE

from tensorflow.keras.models import load_model
model = load_model('D:\\Hackathon_project\\backend\\model.keras')

if st.button("Run Prediction"):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=testing,  
        image_size=(img_height, img_width),
        batch_size=batch_size,  
        shuffle=False  
    )
    
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    l = []
    for images, labels in test_ds:
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        for x in predicted_classes:
            l.append('defected' if x == 0 else 'non-defected')
        
        
    # Display results in Streamlit
    st.write("Predicted Classes:")
    st.write(l)
