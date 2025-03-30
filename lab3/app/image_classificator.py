import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Загружаем модель (кешируем)
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

# Словарь с переводом популярных классов
CLASS_TRANSLATION = {
    "dog": "собака",
    "cat": "кошка",
    "car": "автомобиль",
    "bird": "птица",
    "apple": "яблоко",
    # Добавьте другие нужные переводы
}

def preprocess_image(img):
    # Конвертируем RGBA в RGB при необходимости
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
    st.write("🔍 **Результаты распознавания:**")
    top_indices = preds[0].argsort()[-top:][::-1]
    for i, idx in enumerate(top_indices):
        st.success(f"{i+1}. Class {idx} (probability: {preds[0][idx]*100:.1f}%)")

# --- Основной интерфейс ---
model = load_model()
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