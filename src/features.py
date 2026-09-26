import pandas as pd


def make_features(
    df: pd.DataFrame,
    target: str = "guests"
) -> pd.DataFrame:
    """
    Generate time series features for one or several restaurants.

    All lag and rolling features are shifted so that
    row for day D contains only information known at day D.
    `revenue` is NOT used as a feature to avoid leakage:
    future revenue is unknown at prediction time.

    Used both at training and at inference.
    """
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("Column 'date' must be datetime64[ns]")

    df = df.copy()
    df = (
        df.sort_values(["restaurant_id", "date"])
        .reset_index(drop=True)
    )

    # --- Calendar features ---
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # --- Lags for target ---
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = (
            df.groupby("restaurant_id")[target].shift(lag)
        )

    # --- Rolling statistics for target ---
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
