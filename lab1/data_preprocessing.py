import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(train_folder, test_folder):
    train_data = load_data(train_folder)
    test_data = load_data(test_folder)
    combined_data = pd.concat([df for _, df in train_data + test_data], ignore_index=True)
    scaler = StandardScaler()

    numeric_cols = combined_data.select_dtypes(include=['number']).columns
    scaler.fit(combined_data[numeric_cols])

    for filename, df in train_data:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        save_preprocessed_data(df, train_folder, filename)

    for filename, df in test_data:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        save_preprocessed_data(df, test_folder, filename)

    print("Данные успешно предобработаны и сохранены.")

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

def save_preprocessed_data(df, folder, filename):
    df.to_csv(os.path.join(folder, filename), index=False)

def main():
    train_folder = "train"
    test_folder = "test"
    preprocess_data(train_folder, test_folder)

if __name__ == "__main__":
    main()