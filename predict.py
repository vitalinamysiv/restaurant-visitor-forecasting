import argparse

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
)

from src.features import (
    make_features,
    MODEL_FEATURES,
    is_state_holiday,
    is_school_holiday,
)
from src.model import load_model


def evaluate_on_history(
    df: pd.DataFrame,
    restaurant_id: int,
    model,
    eval_days: int = 14,
) -> dict:
    """
    Walk-forward оценка качества модели на последних `eval_days` днях истории.

    Для каждого дня из хвоста истории делаем прогноз, используя только
    данные до этого дня, и сравниваем с фактом. Так эмулируется реальный
    режим инференса — без утечки будущего.
    """
    df = df[df["restaurant_id"] == restaurant_id].copy()
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < eval_days + 28:
        raise ValueError(
            f"Недостаточно истории для оценки: нужно минимум "
            f"{eval_days + 28} дней, доступно {len(df)}."
        )

    eval_start_idx = len(df) - eval_days
    y_true = []
    y_pred = []

    for idx in range(eval_start_idx, len(df)):
        current_date = df.loc[idx, "date"]
        history = df.iloc[:idx].copy()

        if history.empty:
            continue

        last_revenue = float(history["revenue"].iloc[-1])

        future_state_holiday = is_state_holiday(current_date)
        future_school_holiday = is_school_holiday(current_date)

        future_row = pd.DataFrame(
            {
                "date": [current_date],
                "restaurant_id": [restaurant_id],
                "revenue": [last_revenue],
                "guests": [np.nan],
                "is_state_holiday": [future_state_holiday],
                "is_school_holiday": [future_school_holiday],
                "is_promo": [0],
            }
        )

        temp_df = pd.concat([history, future_row], ignore_index=True)
        temp_features = make_features(temp_df)
        latest = temp_features.tail(1)

        missing = [f for f in MODEL_FEATURES if f not in latest.columns]
        if missing:
            raise ValueError(
                f"В данных отсутствуют признаки, нужные модели: {missing}. "
                f"Проверьте функцию make_features в src/features.py."
            )

        if latest[MODEL_FEATURES].isna().any(axis=1).iloc[0]:
            recent = temp_features[MODEL_FEATURES].tail(60)
            fill_values = recent.median(numeric_only=True)
            latest = latest.copy()
            latest[MODEL_FEATURES] = latest[MODEL_FEATURES].fillna(fill_values)
            latest[MODEL_FEATURES] = latest[MODEL_FEATURES].fillna(0)

        X = latest[MODEL_FEATURES]
        prediction = float(model.predict(X)[0])
        prediction = max(0.0, prediction)

        y_true.append(float(df.loc[idx, "guests"]))
        y_pred.append(prediction)

    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        "MAE": mae,
        "MAPE": mape,
        "n_days": len(y_true),
    }


def predict(
    start_date: str,
    restaurant_id: int,
    model_path: str = "models/catboost_model.pkl",
    data_path: str = "data/processed/restaurant_daily.csv",
    horizon: int = 7,
) -> pd.DataFrame:

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

    df["revenue"] = df["revenue"].fillna(df["revenue"].median())
    last_revenue = float(df["revenue"].iloc[-1])

    for _ in range(horizon):

        future_state_holiday = is_state_holiday(current_date)
        future_school_holiday = is_school_holiday(current_date)

        future_row = pd.DataFrame(
            {
                "date": [current_date],
                "restaurant_id": [restaurant_id],
                "revenue": [last_revenue],
                "guests": [np.nan],
                "is_state_holiday": [future_state_holiday],
                "is_school_holiday": [future_school_holiday],
                "is_promo": [0],
            }
        )

        temp_df = pd.concat([df, future_row], ignore_index=True)
        temp_features = make_features(temp_df)
        latest = temp_features.tail(1)

        missing = [f for f in MODEL_FEATURES if f not in latest.columns]
        if missing:
            raise ValueError(
                f"В данных отсутствуют признаки, нужные модели: {missing}. "
                f"Проверьте функцию make_features в src/features.py."
            )

        if latest[MODEL_FEATURES].isna().any(axis=1).iloc[0]:
            recent = temp_features[MODEL_FEATURES].tail(60)
            fill_values = recent.median(numeric_only=True)
            latest = latest.copy()
            latest[MODEL_FEATURES] = latest[MODEL_FEATURES].fillna(fill_values)
            latest[MODEL_FEATURES] = latest[MODEL_FEATURES].fillna(0)

        X = latest[MODEL_FEATURES]

        prediction = float(model.predict(X)[0])
        prediction = max(0.0, prediction)

        predictions.append(
            {
                "date": current_date.date(),
                "prediction": round(prediction),
            }
        )

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "date": [current_date],
                        "restaurant_id": [restaurant_id],
                        "revenue": [last_revenue],
                        "guests": [prediction],
                        "is_state_holiday": [future_state_holiday],
                        "is_school_holiday": [future_school_holiday],
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
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=True,
        help="Посчитать MAE/MAPE на последних днях истории (по умолчанию включено)",
    )
    parser.add_argument(
        "--no-evaluate",
        dest="evaluate",
        action="store_false",
        help="Отключить оценку качества",
    )
    parser.add_argument(
        "--eval-days",
        type=int,
        default=14,
        help="Сколько последних дней истории использовать для оценки (по умолчанию 14)",
    )

    args = parser.parse_args()

    result = predict(
        args.date,
        args.restaurant,
        model_path=args.model,
        data_path=args.data,
    )

    print(result.to_string(index=False))

    if args.evaluate:
        # Walk-forward оценка на хвосте истории — там, где есть факт.
        # Дата начала прогноза здесь не важна: берём последние eval_days
        # дней из данных ресторана.
        hist_df = pd.read_csv(args.data, parse_dates=["date"])
        model = load_model(args.model)

        try:
            metrics = evaluate_on_history(
                hist_df,
                restaurant_id=args.restaurant,
                model=model,
                eval_days=args.eval_days,
            )
            print()
            print(
                f"Качество на последних {metrics['n_days']} днях истории "
                f"(walk-forward):"
            )
            print(f"  MAE:  {metrics['MAE']:.2f} гостей")
            print(f"  MAPE: {metrics['MAPE']:.2%}")
        except ValueError as e:
            print()
            print(f"[warn] Оценка качества недоступна: {e}")
