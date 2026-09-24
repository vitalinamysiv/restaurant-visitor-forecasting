import argparse
import pandas as pd

from src.features import make_features
from src.model import load_model


def predict(
    date: str,
    restaurant_id: int
):

    # Загружаем данные
    df = pd.read_csv(
        "data/processed/restaurant_daily.csv",
        parse_dates=["date"]
    )


    # Оставляем выбранную точку

    df = df[
        df["restaurant_id"] == restaurant_id
    ].copy()


    if df.empty:
        raise ValueError(
            f"Restaurant {restaurant_id} not found"
        )


    # Добавляем признаки

    df_features = make_features(
        df
    )


    # Берём последние данные

    latest = (
        df_features
        .dropna()
        .tail(1)
    )


    if latest.empty:
        raise ValueError(
            "Not enough history for prediction"
        )


    features = [
        "restaurant_id",
        "revenue",
        "day_of_week",
        "month",
        "day_of_month",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_28"
    ]


    X = latest[features]


    model = load_model()


    prediction = model.predict(
        X
    )


    return prediction[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--date",
        required=True
    )


    parser.add_argument(
        "--restaurant",
        required=True,
        type=int
    )


    args = parser.parse_args()


    result = predict(
        args.date,
        args.restaurant
    )


    print(
        f"Prediction for restaurant {args.restaurant}: {round(result)} guests"
    )
