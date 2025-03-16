import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from pathlib import Path

TRAIN_PATH = "./data/train.csv"
MODEL_PATH = "./model/titanic_model.pkl"
Y_TRAIN_PATH = "./data/y_train.csv"


def train_model():
    try:
        Path("./model").mkdir(parents=True, exist_ok=True)
        x_train = pd.read_csv(TRAIN_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH)
        model = LogisticRegression(solver='liblinear', random_state=42)
        model.fit(x_train, y_train.values.ravel())
        pickle.dump(model, open(MODEL_PATH, 'wb'))
        print(f"Модель обучена и сохранена в {MODEL_PATH}")
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")

if __name__ == "__main__":
    train_model()
