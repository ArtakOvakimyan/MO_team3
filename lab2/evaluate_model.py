import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

TEST_PATH = "./data/test.csv"
MODEL_PATH = "./model/titanic_model.pkl"
Y_TEST_PATH = "./data/y_test.csv"

def evaluate_model():
    try:
        x_test = pd.read_csv(TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH)

        model = pickle.load(open(MODEL_PATH, 'rb'))
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Точность модели на тестовых данных: {accuracy:.4f}")

    except Exception as e:
        print(f"Ошибка при оценке модели: {e}")

if __name__ == "__main__":
    evaluate_model()
