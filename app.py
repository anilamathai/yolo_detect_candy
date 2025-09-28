import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load your trained YOLO model
model = YOLO('my_model/my_model.pt')

st.title("YOLO Object Detection")

option = st.radio("Choose input type:", ("Upload Image", "Webcam"))

# Function to draw bounding boxes
def draw_boxes(frame, results):
    colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), 
              (88,159,106), (96,202,231), (159,124,168), (169,162,241), 
              (98,118,150), (172,176,184)]
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0].cpu().numpy())
        color = colors[cls_id % 10]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        label = model.names[cls_id]
        cv2.putText(frame, label, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        results = model(img)
        img_with_boxes = draw_boxes(img, results)
        st.image(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))

elif option == "Webcam":
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        frame = draw_boxes(frame, results)
        FRAME_WINDOW.image(frame, channels="BGR")
