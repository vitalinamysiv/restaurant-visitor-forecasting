import os
import pandas as pd

from src.features import is_state_holiday, is_school_holiday


def create_daily_dataset(
    input_path: str,
    output_path: str
) -> None:
    """Готовит дневной датасет Rossmann для прогнозирования потока гостей."""
    print("Загрузка данных...")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Не найден файл: {input_path}\n"
            f"Скачайте датасет Rossmann Store Sales "
            f"и положите train.csv в data/raw/train.csv"
        )

    df = pd.read_csv(
        input_path,
        usecols=[
            "Date", "Store", "Customers", "Sales",
            "Open", "Promo", "StateHoliday", "SchoolHoliday",
        ],
        parse_dates=["Date"],
        dtype={"StateHoliday": str},
        low_memory=False,
    )

    print("Размер исходных данных:", df.shape)

    df["is_closed"] = (df["Open"] == 0).astype(int)
    df["is_anomaly"] = ((df["Open"] == 1) & (df["Sales"] == 0)).astype(int)

    n_closed = int(df["is_closed"].sum())
    n_anomaly = int(df["is_anomaly"].sum())
    print(f"Закрытых дней (Open == 0): {n_closed}")
    print(f"Аномалий (Sales == 0 при Open == 1): {n_anomaly}")

    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    df = df.rename(
        columns={
            "Date": "date",
            "Store": "restaurant_id",
            "Customers": "guests",
            "Sales": "revenue",
        }
    )

    df["is_promo"] = df["Promo"].astype(int)
    df["is_state_holiday"] = df["date"].apply(is_state_holiday)
    df["is_school_holiday"] = df["date"].apply(is_school_holiday)

    df = df.drop(
        columns=[
            "Open", "is_closed", "is_anomaly",
            "StateHoliday", "SchoolHoliday", "Promo",
        ]
    )

    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    full_index = pd.MultiIndex.from_product(
        [df["restaurant_id"].unique(), full_range],
        names=["restaurant_id", "date"],
    )

    df = (
        df.set_index(["restaurant_id", "date"])
        .reindex(full_index)
        .reset_index()
        .sort_values(["restaurant_id", "date"])
        .reset_index(drop=True)
    )

    n_missing_guests = int(df["guests"].isna().sum())
    n_missing_revenue = int(df["revenue"].isna().sum())
    print("Размер после восстановления календаря:", df.shape)
    print(f"Пропусков в guests: {n_missing_guests}")
    print(f"Пропусков в revenue: {n_missing_revenue}")

    df["is_state_holiday"] = df["date"].apply(is_state_holiday)
    df["is_school_holiday"] = df["date"].apply(is_school_holiday)
    df["is_promo"] = df["is_promo"].fillna(0).astype(int)

    before = len(df)
    df = df.dropna(subset=["guests", "revenue"]).reset_index(drop=True)
    after = len(df)
    print(f"Удалено строк с пропусками выгрузки: {before - after}")

    df = df.sort_values(["restaurant_id", "date"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Готово!")
    print("Размер обработанных данных:", df.shape)
    print("Сохранено:", output_path)
    print("Колонки:", list(df.columns))


if __name__ == "__main__":
    create_daily_dataset(
        "data/raw/train.csv",
        "data/processed/restaurant_daily.csv",
    )
