import numpy as np
import pandas as pd
import pytest

from src.features import make_features, FEATURES


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


def test_lag_does_not_look_into_future():
    df = _make_df(10)
    out = make_features(df)
    assert out["lag_1"].iloc[5] == df["guests"].iloc[4]
    assert pd.isna(out["lag_1"].iloc[0])


def test_rolling_does_not_use_current_day():
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
    df = pd.DataFrame(
        {
            "restaurant_id": [1],
            "date": ["2024-01-01"], 
            "guests": [10],
            "revenue": [100],
            "is_state_holiday": [0],
            "is_school_holiday": [0],
            "is_promo": [0],
        }
    )
    with pytest.raises(ValueError):
        make_features(df)
