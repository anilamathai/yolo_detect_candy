import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLO model
model = YOLO('my_model/my_model.pt')

st.title("YOLO Object Detection")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)  # Convert PIL image to NumPy array

    # Run YOLO inference
    results = model(img_array)

    # Draw bounding boxes
    annotated_img = results[0].plot()  # returns image with boxes as NumPy array

    st.image(annotated_img, caption="Detected Objects", use_column_width=True)
