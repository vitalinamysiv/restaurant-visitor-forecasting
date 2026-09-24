import argparse
import pandas as pd

from src.features import make_features
from src.model import load_model


FEATURES = [
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


def predict(
    start_date: str,
    restaurant_id: int
):

    # Загружаем историю

    df = pd.read_csv(
        "data/processed/restaurant_daily.csv",
        parse_dates=["date"]
    )


    df = df[
        df["restaurant_id"] == restaurant_id
    ].copy()


    if df.empty:
        raise ValueError(
            f"Restaurant {restaurant_id} not found"
        )


    model = load_model()


    predictions = []


    current_date = pd.to_datetime(start_date)


    # прогнозируем 7 дней

    for i in range(7):

        future_row = pd.DataFrame(
            {
                "date": [current_date],
                "restaurant_id": [restaurant_id],
                "revenue": [
                    df["revenue"].iloc[-1]
                ],
                "guests": [
                    df["guests"].iloc[-1]
                ]
            }
        )


        temp_df = pd.concat(
            [
                df,
                future_row
            ],
            ignore_index=True
        )


        temp_features = make_features(
            temp_df
        )


        latest = (
            temp_features
            .dropna()
            .tail(1)
        )


        X = latest[FEATURES]


        prediction = model.predict(
            X
        )[0]


        predictions.append(
            {
                "date": current_date.date(),
                "prediction": round(prediction)
            }
        )


        # добавляем прогноз как историю

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "date": [current_date],
                        "restaurant_id": [
                            restaurant_id
                        ],
                        "revenue": [
                            df["revenue"].iloc[-1]
                        ],
                        "guests": [
                            prediction
                        ]
                    }
                )
            ],
            ignore_index=True
        )


        current_date += pd.Timedelta(days=1)


    return pd.DataFrame(predictions)



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


    print(result.to_string(index=False))
