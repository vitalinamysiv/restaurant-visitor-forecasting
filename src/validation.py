import pandas as pd


def time_split(
    df: pd.DataFrame,
    valid_weeks: int = 6
):

    cutoff = (
        df["date"].max()
        -
        pd.Timedelta(
            weeks=valid_weeks
        )
    )


    train = df[
        df["date"] <= cutoff
    ]

    valid = df[
        df["date"] > cutoff
    ]


    return train, valid
