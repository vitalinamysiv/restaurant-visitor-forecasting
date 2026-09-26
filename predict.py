import argparse

import pandas as pd

from src.features import make_features, FEATURES
from src.model import load_model


def predict(
    start_date: str,
    restaurant_id: int,
    model_path: str = "models/catboost_model.pkl",
    data_path: str = "data/processed/restaurant_daily.csv",
    horizon: int = 7,
) -> pd.DataFrame:
    """
    Forecast guests for the next `horizon` days
    for a given restaurant.

    Uses the same `make_features` as training
    to avoid train/inference skew.
    """
    df = pd.read_csv(data_path, parse_dates=["date"])

    df = df[df["restaurant_id"] == restaurant_id].copy()

    if df.empty:
        raise ValueError(
            f"Ресторан {restaurant_id} не найден в данных"
        )

    model = load_model(model_path)

    predictions = []
    current_date = pd.to_datetime(start_date)

    # Последняя известная выручка — прокси для будущей,
    # так как будущая выручка неизвестна.
    last_revenue = float(df["revenue"].iloc[-1])

    for _ in range(horizon):
        future_row = pd.DataFrame(
            {
                "date": [current_date],
                "restaurant_id": [restaurant_id],
                "revenue": [last_revenue],
                # guests — то, что мы прогнозируем.
                # Подставим NaN и заполним после make_features.
                "guests": [float("nan")],
                "is_state_holiday": [0],
                "is_school_holiday": [0],
                "is_promo": [0],
            }
        )

        temp_df = pd.concat(
            [df, future_row],
            ignore_index=True,
        )

        temp_features = make_features(temp_df)

        latest = temp_features.tail(1)

        if latest[FEATURES].isna().any(axis=1).iloc[0]:
            raise ValueError(
                f"Недостаточно истории для прогноза на {current_date.date()}. "
                f"Проверьте, что в данных есть минимум 28 дней до даты прогноза."
            )

        X = latest[FEATURES]

        prediction = float(model.predict(X)[0])
        prediction = max(0.0, prediction)

        predictions.append(
            {
                "date": current_date.date(),
                "prediction": round(prediction),
            }
        )

        # Добавляем прогноз как историю для следующего шага
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "date": [current_date],
                        "restaurant_id": [restaurant_id],
                        "revenue": [last_revenue],
                        "guests": [prediction],
                        "is_state_holiday": [0],
                        "is_school_holiday": [0],
                        "is_promo": [0],
                    }
                ),
            ],
            ignore_index=True,
        )

        current_date += pd.Timedelta(days=1)

    return pd.DataFrame(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Прогноз потока гостей на 7 дней вперёд"
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Дата начала прогноза (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--restaurant",
        required=True,
        type=int,
        help="ID ресторана (restaurant_id)",
    )
    parser.add_argument(
        "--model",
        default="models/catboost_model.pkl",
        help="Путь к файлу модели",
    )
    parser.add_argument(
        "--data",
        default="data/processed/restaurant_daily.csv",
        help="Путь к обработанному датасету",
    )

    args = parser.parse_args()

    result = predict(
        args.date,
        args.restaurant,
        model_path=args.model,
        data_path=args.data,
    )

    print(result.to_string(index=False))
