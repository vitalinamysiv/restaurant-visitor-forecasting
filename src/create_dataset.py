import os
import pandas as pd


def create_daily_dataset(
    input_path: str,
    output_path: str
) -> None:

    print("Загрузка данных...")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"""
Не найден файл: {input_path}

Скачайте датасет Rossmann Store Sales
и положите train.csv в:

data/raw/train.csv
"""
        )

    df = pd.read_csv(
        input_path,
        usecols=[
            "Date",
            "Store",
            "Customers",
            "Sales",
            "Open"
        ],
        parse_dates=["Date"]
    )

    print("Размер исходных данных:", df.shape)

    # Оставляем только дни, когда точка была открыта
    df = df[df["Open"] == 1].copy()

    # Приводим названия к схеме проекта
    df = df.rename(
        columns={
            "Date": "date",
            "Store": "restaurant_id",
            "Customers": "guests",
            "Sales": "revenue"
        }
    )

    df = df.drop(
        columns=["Open"]
    )

    # Создаём папку для обработанных данных
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    # Сортируем данные
    df = df.sort_values(
        ["restaurant_id", "date"]
    ).reset_index(drop=True)

    # Сохраняем результат
    df.to_csv(
        output_path,
        index=False
    )

    print("Готово!")
    print("Размер обработанных данных:", df.shape)
    print("Сохранено:", output_path)



if __name__ == "__main__":

    create_daily_dataset(
        "data/raw/train.csv",
        "data/processed/restaurant_daily.csv"
    )
