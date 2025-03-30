# Используем slim-образ вместо alpine (для совместимости с TensorFlow)
FROM python:3.9-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы
COPY app/ .

CMD ["streamlit", "run", "image_classificator.py"]