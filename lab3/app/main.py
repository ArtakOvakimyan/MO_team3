import streamlit as st

st.title("Мой первый микросервис в Docker")
name = st.text_input("Введите ваше имя")
if name:
    st.success(f"Привет, {name}!")