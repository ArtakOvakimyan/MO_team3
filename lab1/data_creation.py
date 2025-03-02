import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

def create_temperature_data(num_days, base_temp, temp_variation, noise_level, anomaly=False, anomaly_day=None, anomaly_temp_increase=None):
    """
    Аргументы:
        num_days (int): Количество дней.
        base_temp (float): Средняя температура.
        temp_variation (float): Максимальное отклонение температуры от среднего.
        noise_level (float): Уровень шума (стандартное отклонение).
        anomaly (bool): Включать ли аномалию.
        anomaly_day (int): День, когда происходит аномалия (начинается с 0).
        anomaly_temp_increase (float): Величина увеличения температуры во время аномалии.
    Возвращает:
        pandas.DataFrame: DataFrame с данными о температуре и датой.
    """

    dates = pd.date_range(start="2023-01-01", periods=num_days)
    temperatures = base_temp + temp_variation * np.sin(np.linspace(0, 2 * np.pi, num_days)) + np.random.normal(0, noise_level, num_days)

    if anomaly:
        if anomaly_day is None or anomaly_temp_increase is None:
            raise ValueError("Если anomaly=True, anomaly_day и anomaly_temp_increase должны быть указаны.")
        temperatures[anomaly_day] += anomaly_temp_increase

    df = pd.DataFrame({"date": dates, "temperature": temperatures})
    return df

def save_data(df, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    df.to_csv(os.path.join(folder, filename), index=False)


def main():
    # Настройка параметров для генерации данных
    num_days = 365
    base_temp = 20.0
    temp_variation = 10.0
    noise_level = 1.0

    # Создание обучающих данных
    train_data_1 = create_temperature_data(num_days, base_temp, temp_variation, noise_level)
    train_data_2 = create_temperature_data(num_days, base_temp, temp_variation, noise_level * 2)  # Больше шума
    train_data_3 = create_temperature_data(num_days, base_temp, temp_variation, noise_level, anomaly=True, anomaly_day=200, anomaly_temp_increase=10) # Аномалия

    # Создание тестовых данных
    test_data_1 = create_temperature_data(num_days, base_temp, temp_variation, noise_level)
    test_data_2 = create_temperature_data(num_days, base_temp + 5, temp_variation, noise_level) # Другая avg температура
    test_data_3 = create_temperature_data(num_days, base_temp, temp_variation, noise_level, anomaly=True, anomaly_day=300, anomaly_temp_increase=15) # Аномалия

    # Сохранение данных
    save_data(train_data_1, "train", "train_data_1.csv")
    save_data(train_data_2, "train", "train_data_2.csv")
    save_data(train_data_3, "train", "train_data_3.csv")

    save_data(test_data_1, "test", "test_data_1.csv")
    save_data(test_data_2, "test", "test_data_2.csv")
    save_data(test_data_3, "test", "test_data_3.csv")

    print("Данные успешно созданы и сохранены в папках 'train' и 'test'")

if __name__ == "__main__":
    main()