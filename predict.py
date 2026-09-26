import argparse

import pandas as pd
import numpy as np

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

    # Сортируем по дате — это критично для лагов
    df = df.sort_values("date").reset_index(drop=True)

    # Проверяем, что до даты прогноза вообще есть данные
    current_date = pd.to_datetime(start_date)
    history_before = df[df["date"] < current_date]

    if history_before.empty:
        raise ValueError(
            f"Нет данных до {current_date.date()} для ресторана {restaurant_id}. "
            f"Минимальная дата в данных: {df['date'].min().date()}"
        )

    # Проверяем, что период истории покрывает хотя бы 28 календарных дней
    days_of_history = (history_before["date"].max() - history_before["date"].min()).days
    if days_of_history < 28:
        raise ValueError(
            f"Недостаточно истории для прогноза на {current_date.date()}. "
            f"Доступно {days_of_history} календарных дней, нужно минимум 28."
        )

    model = load_model(model_path)

    predictions = []

    # Последняя известная выручка — прокси для будущей,
    # так как будущая выручка неизвестна.
    last_revenue = float(df["revenue"].iloc[-1])

    # Заполняем возможные пропуски в revenue (на всякий случай)
    df["revenue"] = df["revenue"].fillna(df["revenue"].median())

    for _ in range(horizon):
        future_row = pd.DataFrame(
            {
                "date": [current_date],
                "restaurant_id": [restaurant_id],
                "revenue": [last_revenue],
                # guests — то, что мы прогнозируем.
                "guests": [np.nan],
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

        # Вместо падения при NaN — заполняем пропуски
        # медианой по последним доступным значениям признака.
        if latest[FEATURES].isna().any(axis=1).iloc[0]:
            nan_cols = latest[FEATURES].columns[
                latest[FEATURES].isna().any()
            ].tolist()

            print(
                f"[WARN] Для {current_date.date()} отсутствуют значения "
                f"признаков: {nan_cols}. Заполняем медианой по истории."
            )

            # Заполняем NaN медианой по последним 60 дням истории
            recent = temp_features[FEATURES].tail(60)
            fill_values = recent.median(numeric_only=True)

            latest = latest.copy()
            latest[FEATURES] = latest[FEATURES].fillna(fill_values)

            # Если и медиана NaN (все значения пропущены) — ставим 0
            latest[FEATURES] = latest[FEATURES].fillna(0)

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
