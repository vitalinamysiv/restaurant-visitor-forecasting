import pandas as pd


# Календарные праздники РФ

RU_STATE_HOLIDAYS = {
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 23),
    (3, 8),
    (5, 1),
    (5, 9),
    (6, 12),
    (11, 4),
}

RU_SCHOOL_HOLIDAY_PERIODS = [
    ((10, 28), (11, 5)),
    ((12, 28), (1, 10)),
    ((3, 22), (3, 31)),
    ((5, 26), (8, 31)),
]


def is_state_holiday(date: pd.Timestamp) -> int:
    """1, если дата — государственный праздник РФ, иначе 0."""
    return int((date.month, date.day) in RU_STATE_HOLIDAYS)


def is_school_holiday(date: pd.Timestamp) -> int:
    """1, если дата попадает в период школьных каникул, иначе 0."""
    for (start_m, start_d), (end_m, end_d) in RU_SCHOOL_HOLIDAY_PERIODS:
        if start_m <= end_m:
            if (start_m, start_d) <= (date.month, date.day) <= (end_m, end_d):
                return 1
        else:
            if (date.month, date.day) >= (start_m, start_d) or \
               (date.month, date.day) <= (end_m, end_d):
                return 1
    return 0


def make_features(
    df: pd.DataFrame,
    target: str = "guests"
) -> pd.DataFrame:
    """
    Генерирует признаки временного ряда для одного или нескольких ресторанов.

    Все лаги и скользящие статистики сдвинуты так, чтобы строка за день D
    содержала только информацию, известную на день D.
    `revenue` не используется как таргет и не порождает лаги,
    но передаётся модели как признак (масштаб активности точки).
    """
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("Колонка 'date' должна быть типа datetime64[ns]")

    df = df.copy()
    df = (
        df.sort_values(["restaurant_id", "date"])
        .reset_index(drop=True)
    )

    # --- Календарные признаки ---
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    # Выходные в РФ: суббота (5) и воскресенье (6)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # --- Лаги целевой переменной ---
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = (
            df.groupby("restaurant_id")[target].shift(lag)
        )

    # --- Скользящие статистики целевой переменной ---
    for window in [7, 28]:
        df[f"rolling_mean_{window}"] = (
            df.groupby("restaurant_id")[target]
            .shift(1)
            .rolling(window)
            .mean()
        )
        df[f"rolling_std_{window}"] = (
            df.groupby("restaurant_id")[target]
            .shift(1)
            .rolling(window)
            .std()
        )

    return df

# Порядок колонок соответствует model.feature_names_ у CatBoost.
MODEL_FEATURES = [
    "restaurant_id",
    "revenue",
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "is_state_holiday",
    "is_school_holiday",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_28",
]

# Полный набор сгенерированных признаков (для тестов и EDA).
# Модель использует подмножество MODEL_FEATURES.
GENERATED_FEATURES = [
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "is_state_holiday",
    "is_school_holiday",
    "is_promo",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
]
