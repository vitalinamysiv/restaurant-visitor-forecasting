import pandas as pd
import os


def create_daily_dataset(
    input_path: str,
    output_path: str
):

    # читаем CSV с правильным разделителем
    df = pd.read_csv(
        input_path,
        sep="|"
    )


    # переводим дату
    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"]
    )


    # считаем выручку каждой покупки
    df["revenue"] = (
        df["transaction_qty"]
        *
        df["unit_price"]
    )


    # агрегируем день + ресторан

    daily = (
        df
        .groupby(
            [
                "transaction_date",
                "store_id"
            ]
        )
        .agg(
            guests=(
                "transaction_id",
                "nunique"
            ),
            revenue=(
                "revenue",
                "sum"
            )
        )
        .reset_index()
    )


    # переименовываем под требования задания

    daily = daily.rename(
        columns={
            "transaction_date": "date",
            "store_id": "restaurant_id"
        }
    )


    # создаем папку

    os.makedirs(
        "data/processed",
        exist_ok=True
    )


    # сохраняем

    daily.to_csv(
        output_path,
        index=False
    )


if __name__ == "__main__":

    create_daily_dataset(
        "coffee-shop-sales-revenue.csv",
        "data/processed/restaurant_daily.csv"
    )
