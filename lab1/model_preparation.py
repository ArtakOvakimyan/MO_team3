import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

def train_model(train_folder, model_filename="temperature_model.pkl"):
    train_data = load_data(train_folder)
    df_train = pd.concat([df for _, df in train_data], ignore_index=True)
    df_train['date'] = (df_train['date'].str.replace('-','').astype(int) - 20230101)

    X = df_train[['date']]
    y = df_train['temperature']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Mean Squared Error (MSE) на валидационной выборке: {mse}")

    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    print(f"Модель успешно обучена и сохранена в файл: {model_filename}")

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
    train_folder = "train"
    train_model(train_folder)

if __name__ == "__main__":
    main()