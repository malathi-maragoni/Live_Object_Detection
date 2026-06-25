import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image

# Load YOLOv5s pretrained on COCO
@st.cache_resource
def load_model():
    return YOLO("yolov5s.pt")  # automatically downloads pretrained weights

model = load_model()

st.title("🔍 Live Object Detection with YOLOv5 (COCO Dataset)")

source_option = st.sidebar.radio("Select Source", ("Webcam", "Upload Image"))

if source_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        results = model(image)
        # Plot results
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Detected Objects", use_column_width=True)

elif source_option == "Webcam":
    run = st.checkbox("Run Webcam")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture frame")
            break

        results = model(frame)
        res_plotted = results[0].plot()
        FRAME_WINDOW.image(res_plotted)

    cap.release()
