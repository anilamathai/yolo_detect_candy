import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch

# Load your trained YOLO model (replace with your own .pt path)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='my_model/yolo11s.pt', force_reload=True)

st.title("YOLO Object Detection (Streamlit-friendly)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image with PIL
    img = Image.open(uploaded_file).convert("RGB")
    
    # Run YOLO inference
    results = model(img)
    
    # Convert results to pandas DataFrame (bbox info)
    df = results.pandas().xyxy[0]
    
    # Draw bounding boxes manually
    draw = ImageDraw.Draw(img)
    for _, row in df.iterrows():
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        label = f"{row['name']} {row['confidence']*100:.1f}%"
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin - 10), label, fill="red")
    
    st.image(img, caption="Detected Objects", use_column_width=True)
    st.write("Detections:", df[['name', 'confidence']])
