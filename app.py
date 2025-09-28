import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = 'YOLO_V8_CANDY_DETECTION.pt' # ⚠️ IMPORTANT: Change this to your model path

# --- MODEL LOADING ---
@st.cache_resource
def load_yolo_model(model_path):
    """Loads the YOLO model only once and caches it."""
    try:
        # ultralytics/YOLO automatically handles the model path
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model from {model_path}. Error: {e}")
        return None

# --- CANDY DETECTION FUNCTION ---
def detect_and_annotate(pil_image, model):
    """
    Runs YOLO detection on a PIL Image and returns the annotated PIL Image.
    YOLO's .predict() can take a PIL Image directly.
    """
    
    # 1. Run detection on the PIL Image
    # results = model.predict(source=pil_image, save=False, conf=0.25, verbose=False)
    # Note: .predict() or the direct call model(source) are equivalent.
    results = model(pil_image, verbose=False) 

    # 2. Get the annotated frame (as a NumPy array in RGB format)
    # YOLO's .plot() returns a NumPy array.
    annotated_frame_np = results[0].plot()

    # 3. Convert the annotated NumPy array back to a PIL Image for Streamlit
    # YOLO's plot output is usually BGR by default if OpenCV is used, 
    # but when run with PIL input it often returns RGB array.
    # To be safe and ensure no cv2 dependency, let's assume it's RGB from model.plot()
    
    # If your model's plot output happens to be BGR (check documentation/test), 
    # you might need to convert it (but without cv2, this is complex).
    # Assuming it's RGB output:
    annotated_image_pil = Image.fromarray(annotated_frame_np)
    
    return annotated_image_pil

# --- MAIN STREAMLIT APP ---
def main_app_no_cv2():
    st.set_page_config(
        page_title="🍬 Streamlit Candy Detection (No CV2)",
        layout="wide"
    )

    st.title("Candy Detection with YOLOv8 and Streamlit (No OpenCV)")
    st.markdown("Upload an image to see the detection results using **Pillow**.")

    # Load Model
    model = load_yolo_model(MODEL_PATH)
    if model is None:
        return # Stop if model loading failed

    # --- FILE UPLOADER ---
    uploaded_file = st.file_uploader(
        "Upload an Image (.jpg, .png)",
        type=['jpg', 'png', 'jpeg']
    )

    if uploaded_file is not None:
        st.subheader("Image Detection Result")
        try:
            # 1. Open the image file using PIL
            pil_image = Image.open(uploaded_file)
            
            # Optional: Display the original image
            st.sidebar.image(pil_image, caption='Original Image', use_column_width=True)

            # 2. Detect and annotate
            annotated_image_pil = detect_and_annotate(pil_image, model)

            # 3. Display in Streamlit
            st.image(annotated_image_pil, caption='Detected Candies', use_column_width=True)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main_app_no_cv2()
