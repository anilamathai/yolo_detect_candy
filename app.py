import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = 'my_model/my_model.pt'  

# --- MODEL LOADING ---
@st.cache_resource
def load_yolo_model(model_path):
    """Load YOLO model once and cache it."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

# --- CANDY DETECTION FUNCTION ---
def detect_and_annotate(pil_image, model):
    """Detect candies and return a properly RGB PIL image."""
    # Run detection (verbose=False suppresses YOLO messages)
    results = model(pil_image, verbose=False)

    # Get annotated frame (BGR)
    annotated_np_bgr = results[0].plot()

    # Convert BGR → RGB to fix blue tint
    annotated_np_rgb = annotated_np_bgr[..., ::-1]

    # Convert to PIL Image
    annotated_image_pil = Image.fromarray(annotated_np_rgb)
    return annotated_image_pil

# --- STREAMLIT APP ---
def main():
    st.set_page_config(page_title="🍬 Candy Detection", layout="wide")
    st.title("Candy Detection with YOLOv8 (No OpenCV Windows)")
    st.markdown("Upload an image to detect candies (using YOLOv8 and PIL).")

    # Load YOLO model
    model = load_yolo_model(MODEL_PATH)
    if model is None:
        return

    # File uploader
    uploaded_file = st.file_uploader("Upload an Image (.jpg, .png, .jpeg)", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file)
            st.sidebar.image(pil_image, caption="Original Image", use_column_width=True)

            annotated_image = detect_and_annotate(pil_image, model)
            st.image(annotated_image, caption="Detected Candies", use_column_width=True)
        except Exception as e:
            st.error(f"Error during detection: {e}")

if __name__ == "__main__":
    main()
