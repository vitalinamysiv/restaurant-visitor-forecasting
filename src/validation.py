from typing import Tuple

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
)


def time_split(
    df: pd.DataFrame,
    valid_weeks: int = 6
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series into train and validation by date.

    The last `valid_weeks` weeks are used for validation,
    everything before — for training. No shuffling.

    Rationale: we forecast the future, so validation must be
    strictly later in time than training.
    """
    if df.empty:
        raise ValueError("DataFrame пустой")

    cutoff = df["date"].max() - pd.Timedelta(weeks=valid_weeks)

    train = df[df["date"] <= cutoff].copy()
    valid = df[df["date"] > cutoff].copy()

    if train.empty or valid.empty:
        raise ValueError(
            f"Недостаточно истории для разбиения: "
            f"train={len(train)}, valid={len(valid)}"
        )

    return train, valid


def evaluate(
    y_true: pd.Series,
    y_pred: pd.Series
) -> dict:
    """Compute MAE and MAPE on the given arrays."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }
