import pandas as pd
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
DATA_PATH = "./data/titanic.csv"

file_path = Path(DATA_PATH)
file_path.parent.mkdir(parents=True, exist_ok=True)

def download_data():
    try:
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_PATH, index=False)
        print(f"Данные успешно скачаны и сохранены в {DATA_PATH}")
    except Exception as e:
        print(f"Ошибка при скачивании данных: {e}")

if __name__ == "__main__":
    download_data()
