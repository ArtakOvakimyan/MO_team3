import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
DATA_PATH = "lab2/data/titanic.csv"

def download_data():
    """Скачивает данные Titanic и сохраняет их в CSV-файл."""
    try:
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_PATH, index=False)
        print(f"Данные успешно скачаны и сохранены в {DATA_PATH}")
    except Exception as e:
        print(f"Ошибка при скачивании данных: {e}")

if __name__ == "__main__":
    download_data()
