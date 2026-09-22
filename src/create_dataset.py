import pandas as pd


def create_daily_dataset(
    input_path: str,
    output_path: str
):

    df = pd.read_csv(input_path)


    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"]
    )


    df["revenue"] = (
        df["transaction_qty"]
        *
        df["unit_price"]
    )


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


    daily = daily.rename(
        columns={
            "transaction_date": "date",
            "store_id": "restaurant_id"
        }
    )


    daily.to_csv(
        output_path,
        index=False
    )


if __name__ == "__main__":

    create_daily_dataset(
        "coffee_shop_sales.csv",
        "data/processed/restaurant_daily.csv"
    )
