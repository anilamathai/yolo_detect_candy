import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- CONFIGURATION ---
# IMPORTANT: Use the correct path based on your repo structure
MODEL_PATH = 'my_model/themy_model.pt' 

# --- MODEL LOADING ---
@st.cache_resource
def load_yolo_model(model_path):
    """Loads the YOLO model only once and caches it."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model from {model_path}. Error: {e}")
        return None

# --- CANDY DETECTION FUNCTION ---
def detect_and_annotate(pil_image, model):
    """
    Runs YOLO detection on a PIL Image, converts BGR output to RGB, 
    and returns the annotated PIL Image.
    """
    
    # 1. Run detection on the PIL Image
    # Set verbose=False to suppress the "0: 640x..." message in the console/terminal
    results = model(pil_image, verbose=False) 

    # 2. Get the annotated frame (as a NumPy array)
    # This array is often in BGR format due to Ultralytics' internal usage of OpenCV logic.
    annotated_frame_np_bgr = results[0].plot()

    # 3. FIX THE BLUE COLOR: Convert BGR to RGB
    # We use NumPy array slicing [..., ::-1] (or [:, :, ::-1]) to reverse the channel order.
    annotated_frame_np_rgb = annotated_frame_np_bgr[:, :, ::-1]
    
    # 4. Convert the annotated NumPy array (now RGB) back to a PIL Image for Streamlit
    annotated_image_pil = Image.fromarray(annotated_frame_np_rgb)
    
    return annotated_image_pil

# --- MAIN STREAMLIT APP ---
def main_app_no_cv2():
    st.set_page_config(
        page_title="🍬 Streamlit Candy Detection",
        layout="wide"
    )

    st.title("Candy Detection with YOLOv8 and Streamlit")
    st.markdown("Upload an image to see the detection results.")

    # Load Model
    model = load_yolo_model(MODEL_PATH)
    if model is None:
        return 

    # --- FILE UPLOADER ---
    uploaded_file = st.file_uploader(
        "Upload an Image (.jpg, .png)",
        type=['jpg', 'png', 'jpeg']
    )

    if uploaded_file is not None:
        st.subheader("Image Detection Result")
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            
            # Optional: Display the original image
            st.sidebar.image(pil_image, caption='Original Image', use_column_width=True)

            # 2. Detect and annotate
            # Add a temporary spinner to show processing is underway
            with st.spinner('Running detection...'):
                annotated_image_pil = detect_and_annotate(pil_image, model)

            # 3. Display in Streamlit
            st.image(annotated_image_pil, caption='Detected Candies', use_column_width=True)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main_app_no_cv2()
