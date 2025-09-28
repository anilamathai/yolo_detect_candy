import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Candy Detection", layout="wide")
st.title("🍬 Candy Detection App")

# Upload YOLO model
model_file = st.file_uploader("Upload YOLO Model (.pt)", type=["pt"])
if model_file:
    with open("temp_model.pt", "wb") as f:
        f.write(model_file.read())
    model = YOLO("temp_model.pt")
    st.success("Model loaded successfully!")

# Choose source type
source_type = st.radio("Select input type", ["Image", "Video"])

if source_type == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image and model_file:
        image = Image.open(uploaded_image)
        frame = np.array(image)
        results = model(frame)
        annotated_frame = results[0].plot()
        st.image(annotated_frame, caption="Detected Image", use_column_width=True)

elif source_type == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video and model_file:
        tfile = "temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR")
        cap.release()
