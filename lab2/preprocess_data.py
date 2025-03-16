import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "lab2/data/titanic.csv"
TRAIN_PATH = "lab2/data/train.csv"
TEST_PATH = "lab2/data/test.csv"
TARGET = "Survived"

def preprocess_data():
    """
    Загружает данные, выполняет предварительную обработку,
    разделяет на тренировочный и тестовый наборы и сохраняет.
    """
    try:
        df = pd.read_csv(DATA_PATH)

        # 1. Обработка пропущенных значений (пример)
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

        # 2. Кодирование категориальных признаков (пример)
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

        # 3. Выбор признаков (Features Selection)
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        X = df[features]
        y = df[TARGET]

        # 4. Разделение на тренировочный и тестовый наборы
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 5. Сохранение датасетов
        X_train.to_csv(TRAIN_PATH, index=False)
        X_test.to_csv(TEST_PATH, index=False)
        y_train.to_csv("lab2/data/y_train.csv", index=False)
        y_test.to_csv("lab2/data/y_test.csv", index=False)

        print(f"Данные предобработаны и разделены. Train: {TRAIN_PATH}, Test: {TEST_PATH}")

    except Exception as e:
        print(f"Ошибка при предобработке данных: {e}")

if __name__ == "__main__":
    preprocess_data()
