import argparse

import pandas as pd
import numpy as np

from src.features import make_features, FEATURES
from src.model import load_model


# Порядок признаков, который ожидает обученная модель CatBoost.
# Получен через: model.feature_names_
MODEL_FEATURES = [
    'restaurant_id', 'revenue', 'day_of_week', 'month', 'day_of_month',
    'is_weekend', 'lag_1', 'lag_7', 'lag_14',
    'rolling_mean_7', 'rolling_mean_28',
]


def predict(
    start_date: str,
    restaurant_id: int,
    model_path: str = "models/catboost_model.pkl",
    data_path: str = "data/processed/restaurant_daily.csv",
    horizon: int = 7,
) -> pd.DataFrame:
    """
    Forecast guests for the next `horizon` days for a given restaurant.
    Uses the same `make_features` as training to avoid train/inference skew.
    """
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df[df["restaurant_id"] == restaurant_id].copy()

    if df.empty:
        raise ValueError(f"Ресторан {restaurant_id} не найден в данных")

    df = df.sort_values("date").reset_index(drop=True)

    current_date = pd.to_datetime(start_date)
    history_before = df[df["date"] < current_date]

    if history_before.empty:
        raise ValueError(
            f"Нет данных до {current_date.date()} для ресторана {restaurant_id}. "
            f"Минимальная дата в данных: {df['date'].min().date()}"
        )

    days_of_history = (history_before["date"].max() - history_before["date"].min()).days
    if days_of_history < 28:
        raise ValueError(
            f"Недостаточно истории для прогноза на {current_date.date()}. "
            f"Доступно {days_of_history} календарных дней, нужно минимум 28."
        )

    model = load_model(model_path)

    predictions = []
    last_revenue = float(df["revenue"].iloc[-1])
    df["revenue"] = df["revenue"].fillna(df["revenue"].median())

    for _ in range(horizon):
        future_row = pd.DataFrame(
            {
                "date": [current_date],
                "restaurant_id": [restaurant_id],
                "revenue": [last_revenue],
                "guests": [np.nan],
                "is_state_holiday": [0],
                "is_school_holiday": [0],
                "is_promo": [0],
            }
        )

        temp_df = pd.concat([df, future_row], ignore_index=True)
        temp_features = make_features(temp_df)
        latest = temp_features.tail(1)

        # Проверяем, что все признаки модели присутствуют
        missing = [f for f in MODEL_FEATURES if f not in latest.columns]
        if missing:
            raise ValueError(
                f"В данных отсутствуют признаки, нужные модели: {missing}. "
                f"Проверьте функцию make_features в src/features.py."
            )

        # Заполняем NaN медианой по последним 60 дням
        if latest[MODEL_FEATURES].isna().any(axis=1).iloc[0]:
            nan_cols = latest[MODEL_FEATURES].columns[
                latest[MODEL_FEATURES].isna().any()
            ].tolist()
            print(
                f"[WARN] Для {current_date.date()} отсутствуют значения "
                f"признаков: {nan_cols}. Заполняем медианой по истории."
            )
            recent = temp_features[MODEL_FEATURES].tail(60)
            fill_values = recent.median(numeric_only=True)
            latest = latest.copy()
            latest[MODEL_FEATURES] = latest[MODEL_FEATURES].fillna(fill_values)
            latest[MODEL_FEATURES] = latest[MODEL_FEATURES].fillna(0)

        # Приводим порядок колонок к порядку модели
        X = latest[MODEL_FEATURES]

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
        "--date", required=True, help="Дата начала прогноза (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--restaurant", required=True, type=int, help="ID ресторана (restaurant_id)"
    )
    parser.add_argument(
        "--model", default="models/catboost_model.pkl", help="Путь к файлу модели"
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
