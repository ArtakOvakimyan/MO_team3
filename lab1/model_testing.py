import os
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error

def test_model(test_folder, model_filename="temperature_model.pkl"):
    test_data = load_data(test_folder)

    df_test = pd.concat([df for _, df in test_data], ignore_index=True)

    df_test['date'] = (df_test['date'].str.replace('-','').astype(int) - 20230101)

    X_test = df_test[['date']]
    y_test = df_test['temperature']

    try:
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        print(f"Ошибка: Файл модели не найден: {model_filename}")
        return

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Mean Squared Error (MMSE) на тестовых данных: {mse}")

def load_data(folder):
    """
    Возвращает:
        list: Список кортежей (filename, DataFrame), где filename - имя файла, а DataFrame - загруженные данные.
    """
    data = []
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath)
            data.append((filename, df))
    return data

def main():
    test_folder = "test"
    test_model(test_folder)

if __name__ == "__main__":
    main()