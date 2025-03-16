import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

TRAIN_PATH = "lab2/data/train.csv"
MODEL_PATH = "lab2/model/titanic_model.pkl"
Y_TRAIN_PATH = "lab2/data/y_train.csv"


def train_model():
    """Обучает модель логистической регрессии и сохраняет ее."""
    try:
        X_train = pd.read_csv(TRAIN_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH)
        model = LogisticRegression(solver='liblinear', random_state=42)
        model.fit(X_train, y_train.values.ravel()) #Обучение модели
        pickle.dump(model, open(MODEL_PATH, 'wb'))  # Сохранение модели
        print(f"Модель обучена и сохранена в {MODEL_PATH}")
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")

if __name__ == "__main__":
    train_model()
