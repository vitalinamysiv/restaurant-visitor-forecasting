import pandas as pd


def make_features(
    df: pd.DataFrame,
    target: str = "guests"
) -> pd.DataFrame:
    """
    Generate time series features.
    """

    df = df.copy()

    df = df.sort_values(
        ["restaurant_id", "date"]
    )


    # календарные признаки

    df["day_of_week"] = (
        df["date"]
        .dt.dayofweek
    )

    df["month"] = (
        df["date"]
        .dt.month
    )

    df["day_of_month"] = (
        df["date"]
        .dt.day
    )

    df["is_weekend"] = (
        df["day_of_week"]
        >= 5
    ).astype(int)


    # лаги

    for lag in [1, 7, 14]:

        df[f"lag_{lag}"] = (
            df
            .groupby("restaurant_id")[target]
            .shift(lag)
        )


    # rolling признаки

    for window in [7, 28]:

        df[f"rolling_mean_{window}"] = (
            df
            .groupby("restaurant_id")[target]
            .shift(1)
            .rolling(window)
            .mean()
        )


    return df
