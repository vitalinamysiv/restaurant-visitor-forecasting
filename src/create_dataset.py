import pandas as pd
import os


def create_daily_dataset(
    input_path: str,
    output_path: str
):

    # загрузка данных

    df = pd.read_csv(
        input_path,
        parse_dates=["Date"]
    )


    # оставляем только нужные поля

    df = df[
        [
            "Date",
            "Store",
            "Customers",
            "Sales",
            "Open"
        ]
    ]


    # переименовываем под задачу

    df = df.rename(
        columns={
            "Date": "date",
            "Store": "restaurant_id",
            "Customers": "guests",
            "Sales": "revenue"
        }
    )


    # сохраняем только открытые точки

    df = df[
        df["Open"] == 1
    ]


    df = df.drop(
        columns=["Open"]
    )


    os.makedirs(
        "data/processed",
        exist_ok=True
    )


    df.to_csv(
        output_path,
        index=False
    )


if __name__ == "__main__":

    create_daily_dataset(
        "data/raw/train.csv",
        "data/processed/restaurant_daily.csv"
    )
