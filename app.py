import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Load your trained YOLO model
model = YOLO('my_model/my_model.pt')

st.title("YOLO Object Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

# Function to draw bounding boxes on a PIL image
def draw_boxes_pil(pil_img, results):
    draw = ImageDraw.Draw(pil_img)
    colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), 
              (88,159,106), (96,202,231), (159,124,168), (169,162,241), 
              (98,118,150), (172,176,184)]
    
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0].cpu().numpy())
        color = colors[cls_id % 10]
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        label = model.names[cls_id]
        draw.text((xmin, ymin-10), label, fill=color)
    
    return pil_img

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    results = model(np.array(img))
    img_with_boxes = draw_boxes_pil(img, results)
    st.image(img_with_boxes)
