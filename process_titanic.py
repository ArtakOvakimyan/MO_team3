import pandas as pd
import argparse
import sys
import os

def select_columns(input_path, output_path):
    """
    Выбирает колонки 'Pclass', 'Sex', 'Age' из CSV файла.
    """
    try:
        df = pd.read_csv(input_path)
        # Проверяем наличие необходимых колонок
        required_cols = ['Pclass', 'Sex', 'Age']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Ошибка: Отсутствуют необходимые колонки: {missing} в файле {input_path}", file=sys.stderr)
            sys.exit(1)

        df_selected = df[required_cols]
        df_selected.to_csv(output_path, index=False)
        print(f"Выбраны колонки 'Pclass', 'Sex', 'Age'. Результат сохранен в {output_path}")
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {input_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при обработке файла {input_path}: {e}", file=sys.stderr)
        sys.exit(1)

def fill_age(input_path, output_path):
    """
    Заполняет пропущенные значения в колонке 'Age' средним значением.
    """
    try:
        df = pd.read_csv(input_path)
        if 'Age' not in df.columns:
            print(f"Ошибка: Колонка 'Age' отсутствует в файле {input_path}", file=sys.stderr)
            sys.exit(1)

        if df['Age'].isnull().any():
            mean_age = df['Age'].mean()
            df['Age'].fillna(mean_age, inplace=True)
            print(f"Пропущенные значения 'Age' заполнены средним: {mean_age:.2f}")
        else:
            print("В колонке 'Age' нет пропущенных значений.")

        df.to_csv(output_path, index=False)
        print(f"Результат с заполненным 'Age' сохранен в {output_path}")
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {input_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при обработке файла {input_path}: {e}", file=sys.stderr)
        sys.exit(1)

def one_hot_sex(input_path, output_path):
    """
    Применяет one-hot encoding к колонке 'Sex'.
    """
    try:
        df = pd.read_csv(input_path)
        if 'Sex' not in df.columns:
            print(f"Ошибка: Колонка 'Sex' отсутствует в файле {input_path}", file=sys.stderr)
            sys.exit(1)

        # Проверяем, не было ли уже применено one-hot encoding
        if 'Sex_male' in df.columns or 'Sex_female' in df.columns:
             print("Похоже, one-hot encoding для 'Sex' уже применен. Пропуск шага.")
        else:
            df = pd.get_dummies(df, columns=['Sex'], prefix='Sex', drop_first=False) # drop_first=False чтобы были обе колонки
            print("Применен one-hot encoding для колонки 'Sex'.")

        df.to_csv(output_path, index=False)
        print(f"Результат с one-hot encoding для 'Sex' сохранен в {output_path}")

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {input_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при обработке файла {input_path}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обработка датасета Titanic.")
    parser.add_argument('task', choices=['select_columns', 'fill_age', 'one_hot_sex'],
                        help="Задача для выполнения: 'select_columns', 'fill_age', 'one_hot_sex'.")
    parser.add_argument('--input', type=str, required=True, help="Путь к входному CSV файлу.")
    parser.add_argument('--output', type=str, required=True, help="Путь к выходному CSV файлу.")

    args = parser.parse_args()

    # Убедимся, что директория для выходного файла существует
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.task == 'select_columns':
        select_columns(args.input, args.output)
    elif args.task == 'fill_age':
        fill_age(args.input, args.output)
    elif args.task == 'one_hot_sex':
        one_hot_sex(args.input, args.output) 