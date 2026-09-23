import os
import pandas as pd


def create_daily_dataset(
    input_path: str,
    output_path: str
) -> None:
    """
    Create a daily dataset for restaurant visitor forecasting.

    Parameters
    ----------
    input_path : str
        Path to the raw Rossmann train.csv.
    output_path : str
        Path where the processed dataset will be saved.
    """

    print("Загрузка данных...")

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

    # Приводим названия к схеме нашего проекта
    df = df.rename(
        columns={
            "Date": "date",
            "Store": "restaurant_id",
            "Customers": "guests",
            "Sales": "revenue"
        }
    )

    df = df.drop(columns=["Open"])

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
