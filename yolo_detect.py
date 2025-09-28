
import cv2
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model (.pt file)")
    parser.add_argument("--source", type=str, required=True, help="Video source: webcam index (0,1,2) or usb0/usb1, or path to video/image")
    parser.add_argument("--resolution", type=str, default="640x480", help="Resolution WxH, e.g., 1280x720")
    args = parser.parse_args()

    # Load YOLO model
    model = YOLO(args.model)

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split("x"))
    except:
        width, height = 640, 480

    # Detect source type
    img_source = args.source

    if img_source.isdigit():
        # Directly a webcam index
        cap = cv2.VideoCapture(int(img_source))
    elif img_source.startswith("usb") and img_source[3:].isdigit():
        # usb0 → webcam 0
        cap = cv2.VideoCapture(int(img_source[3:]))
    else:
        # Treat as file path
        cap = cv2.VideoCapture(img_source)

    if not cap.isOpened():
        print(f"❌ Could not open source: {img_source}")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        # Draw detections
        annotated_frame = results[0].plot()

        cv2.imshow("YOLO Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
