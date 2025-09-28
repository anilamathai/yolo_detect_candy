import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load model
model = YOLO("my_model/my_model.pt")  # your trained model

st.title("YOLO Object Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    # Run inference
    results = model(img_array)

    # Draw boxes on image
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        label = model.names[cls_id]
        conf = box.conf.item()
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        # Draw rectangle manually using PIL
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle(list(xyxy), outline="red", width=2)
        draw.text((xyxy[0], xyxy[1]-10), f"{label} {conf:.2f}", fill="red")

    st.image(img, caption="Detected objects", use_column_width=True)
