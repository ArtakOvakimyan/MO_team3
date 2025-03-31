import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import requests
import json

@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')


@st.cache_data
def load_imagenet_labels():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    response = requests.get(url)
    class_data = response.json()
    return {int(idx): (class_id, name) for idx, (class_id, name) in class_data.items()}

def preprocess_image(img):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

def load_image():
    uploaded_file = st.file_uploader("📤 Загрузите изображение", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Ваше изображение", use_column_width=True)
        return img
    return None

def display_predictions(preds, top=3):
    decoded_preds = decode_predictions(preds, top=top)[0]
    
    st.write("🔍 **Результаты распознавания:**")
    for i, (class_id, class_name_en, prob) in enumerate(decoded_preds):
        st.success(f"{i+1}. {class_name_en} (точность: {prob*100:.1f}%)")

# --- Основной интерфейс ---
model = load_model()
imagenet_labels = load_imagenet_labels()
st.title("🖼️ Классификатор изображений")

img = load_image()
if img and st.button("🔎 Распознать"):
    with st.spinner("Анализируем изображение..."):
        try:
            x = preprocess_image(img)
            preds = model.predict(x)
            display_predictions(preds)
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")