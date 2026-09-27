import pandas as pd


# ============================================================
# Календарные праздники РФ
# ============================================================

# Государственные праздники (фиксированные даты, формат (месяц, день))
RU_STATE_HOLIDAYS = {
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),  # Новогодние каникулы
    (2, 23),   # День защитника Отечества
    (3, 8),    # Международный женский день
    (5, 1),    # Праздник Весны и Труда
    (5, 9),    # День Победы
    (6, 12),   # День России
    (11, 4),   # День народного единства
}

# Периоды школьных каникул: ((месяц_начала, день_начала), (месяц_конца, день_конца))
# Упрощённые периоды; при необходимости легко уточнить.
RU_SCHOOL_HOLIDAY_PERIODS = [
    ((10, 28), (11, 5)),    # Осенние каникулы
    ((12, 28), (1, 10)),    # Зимние каникулы (переходят через год)
    ((3, 22), (3, 31)),     # Весенние каникулы
    ((5, 26), (8, 31)),     # Летние каникулы
]


def is_state_holiday(date: pd.Timestamp) -> int:
    """
    Возвращает 1, если дата — государственный праздник РФ, иначе 0.

    Работает для любой даты, включая будущие, — праздники известны заранее.
    Это позволяет корректно прогнозировать поток в праздничные дни,
    не «подглядывая» в будущее.
    """
    return int((date.month, date.day) in RU_STATE_HOLIDAYS)


def is_school_holiday(date: pd.Timestamp) -> int:
    """
    Возвращает 1, если дата попадает в период школьных каникул, иначе 0.

    Работает для любой даты, включая будущие, — каникулы известны заранее.
    Корректно обрабатывает периоды, переходящие через Новый год
    (например, зимние каникулы с 28 декабря по 10 января).
    """
    for (start_m, start_d), (end_m, end_d) in RU_SCHOOL_HOLIDAY_PERIODS:
        if start_m <= end_m:
            # Обычный период внутри одного года
            if (start_m, start_d) <= (date.month, date.day) <= (end_m, end_d):
                return 1
        else:
            # Период переходит через Новый год (декабрь -> январь)
            if (date.month, date.day) >= (start_m, start_d) or \
               (date.month, date.day) <= (end_m, end_d):
                return 1
    return 0


# ============================================================
# Генерация признаков
# ============================================================

def make_features(
    df: pd.DataFrame,
    target: str = "guests"
) -> pd.DataFrame:
    """
    Генерирует признаки временного ряда для одного или нескольких ресторанов.

    Все лаги и скользящие статистики сдвинуты так, чтобы строка за день D
    содержала только информацию, известную на день D.
    `revenue` НЕ используется как признак, чтобы избежать утечки:
    будущая выручка неизвестна в момент прогноза.

    Используется и при обучении, и при инференсе.
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


FEATURES = [
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
