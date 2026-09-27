import numpy as np
import pandas as pd
import pytest

from src.features import (
    make_features,
    FEATURES,
    is_state_holiday,
    is_school_holiday,
)

# Хелпер: синтетический дневной ряд для одной точки

def _make_df(n: int = 40) -> pd.DataFrame:
    """Синтетический дневной ряд для одной точки."""
    return pd.DataFrame(
        {
            "restaurant_id": [1] * n,
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "guests": np.arange(n, dtype=float),
            "revenue": np.arange(n, dtype=float) * 100,
            "is_state_holiday": [0] * n,
            "is_school_holiday": [0] * n,
            "is_promo": [0] * n,
        }
    )

# Тесты на функции праздников

def test_state_holiday_known_dates():
    """1 января и 9 мая — государственные праздники."""
    assert is_state_holiday(pd.Timestamp("2024-01-01")) == 1
    assert is_state_holiday(pd.Timestamp("2024-05-09")) == 1
    assert is_state_holiday(pd.Timestamp("2024-03-15")) == 0


def test_state_holiday_works_for_future():
    """Функция работает для будущих дат (праздники известны заранее)."""
    assert is_state_holiday(pd.Timestamp("2030-01-01")) == 1
    assert is_state_holiday(pd.Timestamp("2030-06-01")) == 0


def test_school_holiday_summer():
    """Июль — летние каникулы."""
    assert is_school_holiday(pd.Timestamp("2024-07-15")) == 1
    assert is_school_holiday(pd.Timestamp("2024-10-15")) == 0


def test_school_holiday_winter_crosses_year():
    """Зимние каникулы переходят через год."""
    assert is_school_holiday(pd.Timestamp("2024-12-30")) == 1
    assert is_school_holiday(pd.Timestamp("2025-01-05")) == 1
    assert is_school_holiday(pd.Timestamp("2025-01-15")) == 0

# Тесты на make_features

def test_lag_does_not_look_into_future():
    """Lag_1 не должен подглядывать в будущее."""
    df = _make_df(10)
    out = make_features(df)
    assert out["lag_1"].iloc[5] == df["guests"].iloc[4]
    assert pd.isna(out["lag_1"].iloc[0])


def test_rolling_does_not_use_current_day():
    """Rolling_mean_7 использует shift(1), поэтому не видит текущий день."""
    df = _make_df(40)
    out = make_features(df)
    # Первые 7 значений rolling_mean_7 должны быть NaN,
    # так как используется shift(1) + rolling(7).
    assert out["rolling_mean_7"].iloc[:7].isna().all()
    # На позиции 7 rolling_mean_7 = среднее guests[0:7]
    expected = df["guests"].iloc[0:7].mean()
    assert out["rolling_mean_7"].iloc[7] == pytest.approx(expected)


def test_revenue_not_in_features():
    """FEATURES не должен содержать revenue (утечка)."""
    assert "revenue" not in FEATURES
    assert "guests" not in FEATURES


def test_missing_date_raises():
    """make_features должен падать с ValueError, если date не datetime."""
    df = pd.DataFrame(
        {
            "restaurant_id": [1],
            "date": ["2024-01-01"],  # ← строка, а не Timestamp
            "guests": [10],
            "revenue": [100],
            "is_state_holiday": [0],
            "is_school_holiday": [0],
            "is_promo": [0],
        }
    )
    with pytest.raises(ValueError):
        make_features(df)
