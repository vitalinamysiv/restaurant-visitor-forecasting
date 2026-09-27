import numpy as np
import pandas as pd
import pytest

from src.features import (
    make_features,
    MODEL_FEATURES,
    GENERATED_FEATURES,
    is_state_holiday,
    is_school_holiday,
)


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


# --- Праздники ---

def test_state_holiday_known_dates():
    assert is_state_holiday(pd.Timestamp("2024-01-01")) == 1
    assert is_state_holiday(pd.Timestamp("2024-05-09")) == 1
    assert is_state_holiday(pd.Timestamp("2024-03-15")) == 0


def test_state_holiday_works_for_future():
    assert is_state_holiday(pd.Timestamp("2030-01-01")) == 1
    assert is_state_holiday(pd.Timestamp("2030-06-01")) == 0


def test_school_holiday_summer():
    assert is_school_holiday(pd.Timestamp("2024-07-15")) == 1
    assert is_school_holiday(pd.Timestamp("2024-10-15")) == 0


def test_school_holiday_winter_crosses_year():
    assert is_school_holiday(pd.Timestamp("2024-12-30")) == 1
    assert is_school_holiday(pd.Timestamp("2025-01-05")) == 1
    assert is_school_holiday(pd.Timestamp("2025-01-15")) == 0


# --- Базовые проверки лагов и rolling ---

def test_lag_does_not_look_into_future():
    df = _make_df(10)
    out = make_features(df)
    assert out["lag_1"].iloc[5] == df["guests"].iloc[4]
    assert pd.isna(out["lag_1"].iloc[0])


def test_rolling_does_not_use_current_day():
    df = _make_df(40)
    out = make_features(df)
    assert out["rolling_mean_7"].iloc[:7].isna().all()
    expected = df["guests"].iloc[0:7].mean()
    assert out["rolling_mean_7"].iloc[7] == pytest.approx(expected)

# Меняем будущие значения и убеждаемся, что признаки прошлых строк не изменились.
def test_no_future_leakage_on_all_features():
    """Изменение будущего не должно влиять на признаки прошлого."""
    df = _make_df(60)
    out_before = make_features(df)

    df_future_changed = df.copy()
    # Меняем значения строго после индекса 30
    df_future_changed.loc[31:, "guests"] = 9999.0
    df_future_changed.loc[31:, "revenue"] = 999999.0
    out_after = make_features(df_future_changed)

    # Все признаки для строк 0..30 должны совпасть
    leakage_cols = [
        "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_28",
        "rolling_std_7", "rolling_std_28",
    ]
    pd.testing.assert_frame_equal(
        out_before.loc[:30, leakage_cols].reset_index(drop=True),
        out_after.loc[:30, leakage_cols].reset_index(drop=True),
        check_dtype=False,
    )


# --- Контракт признаков ---

def test_model_features_subset_of_generated():
    """Все признаки модели должны генерироваться make_features."""
    df = _make_df(40)
    out = make_features(df)
    for feat in MODEL_FEATURES:
        assert feat in out.columns, f"Признак {feat} не генерируется"


def test_generated_features_all_present():
    """GENERATED_FEATURES совпадает с реально сгенерированными (кроме id/date/target)."""
    df = _make_df(40)
    out = make_features(df)
    for feat in GENERATED_FEATURES:
        assert feat in out.columns, f"Признак {feat} не генерируется"


def test_target_not_in_features():
    """Целевая переменная не должна быть признаком."""
    assert "guests" not in MODEL_FEATURES
    assert "guests" not in GENERATED_FEATURES


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
