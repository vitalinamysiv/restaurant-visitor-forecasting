import os
import pandas as pd


def create_daily_dataset(
    input_path: str,
    output_path: str
) -> None:
    """
    Prepare daily dataset for restaurant visitor forecasting
    from Rossmann Store Sales.

    Steps:
    - load raw data with required columns;
    - mark closed days and anomalies;
    - keep only open days without anomalies;
    - rename columns to project schema;
    - restore calendar explicitly (reindex by full date range);
    - save processed dataset.
    """
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
            "Date",
            "Store",
            "Customers",
            "Sales",
            "Open",
            "Promo",
            "StateHoliday",
            "SchoolHoliday",
        ],
        parse_dates=["Date"],
    )

    print("Размер исходных данных:", df.shape)

    # Помечаем закрытые дни и аномалии (Sales == 0 при Open == 1)
    df["is_closed"] = (df["Open"] == 0).astype(int)
    df["is_anomaly"] = (
        (df["Open"] == 1) & (df["Sales"] == 0)
    ).astype(int)

    n_closed = int(df["is_closed"].sum())
    n_anomaly = int(df["is_anomaly"].sum())
    print(f"Закрытых дней (Open == 0): {n_closed}")
    print(f"Аномалий (Sales == 0 при Open == 1): {n_anomaly}")

    # Оставляем только открытые дни без аномалий
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    # Приводим названия к схеме проекта
    df = df.rename(
        columns={
            "Date": "date",
            "Store": "restaurant_id",
            "Customers": "guests",
            "Sales": "revenue",
        }
    )

    # StateHoliday может быть числом (0) или строкой ('a', 'b', 'c')
    df["StateHoliday"] = df["StateHoliday"].astype(str)
    df["is_state_holiday"] = (
        df["StateHoliday"].isin(["a", "b", "c"])
    ).astype(int)
    df["is_school_holiday"] = df["SchoolHoliday"].astype(int)
    df["is_promo"] = df["Promo"].astype(int)

    df = df.drop(
        columns=[
            "Open",
            "is_closed",
            "is_anomaly",
            "StateHoliday",
            "SchoolHoliday",
            "Promo",
        ]
    )

    # Восстанавливаем календарь: для каждой точки — все дни
    # между минимальной и максимальной датой в датасете.
    # Пропуски остаются как NaN, чтобы отличать их от закрытых дней.
    full_range = pd.date_range(
        df["date"].min(),
        df["date"].max(),
        freq="D",
    )
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

    # Удаляем строки с пропусками выгрузки (не закрытые дни,
    # а именно отсутствующие наблюдения). Логируем, сколько удалили.
    before = len(df)
    df = df.dropna(
        subset=["guests", "revenue"]
    ).reset_index(drop=True)
    after = len(df)
    print(f"Удалено строк с пропусками выгрузки: {before - after}")

    # Флаги праздников и промо заполняем 0 там, где NaN
    for col in ["is_state_holiday", "is_school_holiday", "is_promo"]:
        df[col] = df[col].fillna(0).astype(int)

    df = df.sort_values(
        ["restaurant_id", "date"]
    ).reset_index(drop=True)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

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
