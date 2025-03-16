import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "./data/titanic.csv"
TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"
TARGET = "Survived"

def preprocess_data():
    try:
        df = pd.read_csv(DATA_PATH)

        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        x = df[features]
        y = df[TARGET]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        x_train.to_csv(TRAIN_PATH, index=False)
        x_test.to_csv(TEST_PATH, index=False)
        y_train.to_csv("./data/y_train.csv", index=False)
        y_test.to_csv("./data/y_test.csv", index=False)

        print(f"Данные предобработаны и разделены. Train: {TRAIN_PATH}, Test: {TEST_PATH}")

    except Exception as e:
        print(f"Ошибка при предобработке данных: {e}")

if __name__ == "__main__":
    preprocess_data()