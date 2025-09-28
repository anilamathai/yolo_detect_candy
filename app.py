import streamlit as st
import torch
from PIL import Image
import numpy as np

# Load your trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='my_model/my_model.pt', force_reload=True)

st.title("YOLO Object Detection")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    results = model(np.array(img))  # Run detection
    st.image(np.squeeze(results.render()), caption="Detected Objects", use_column_width=True)
