import streamlit as st
from PIL import Image
import base64
import io
from utils import resize_and_convert_to_base64
from LLM import generate_response_openai
from yolo_pred import test_trained_model

st.set_page_config(page_title="PCB Defect Classifier", layout="centered")

st.title("🔍 PCB Defect Classification with YOLO + LLM Captioning")

uploaded_file = st.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    
    best_model_path = "/runs/classify/pcb_defects_v1/weights/best.pt"


    b64_img = resize_and_convert_to_base64(uploaded_file, max_width=512)


    predicted_class, confidence = test_trained_model(model_path=best_model_path, uploaded_file= uploaded_file)
    prompt = f"Given an image of a PCB defect, predicted the defect is: {predicted_class},  with a confidence score of {confidence:.2f}. Write a concise caption describing this defect in a factual tone."


    caption = generate_response_openai(image_b64= b64_img, prompt=prompt)
    st.markdown(f"### 📝 LLM Caption:\n> {caption}")
