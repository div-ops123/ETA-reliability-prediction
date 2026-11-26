# src/data_modeling/data_cleaning.py

import pandas as pd


RAW_COLUMNS = {
    "market_id",
    "created_at",
    "actual_delivery_time",
    "store_id",
    "store_primary_category",
    "order_protocol",
    "total_items",
    "subtotal",
    "num_distinct_items",
    "min_item_price",
    "max_item_price",
    "total_onshift_dashers",
    "total_busy_dashers",
    "total_outstanding_orders",
    "estimated_order_place_duration",
    "estimated_store_to_consumer_driving_duration",
}


def load_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing_columns = RAW_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")


def parse_timestamps(
    df: pd.DataFrame,
    date_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()

    for col in date_cols:
        df[col] = pd.to_datetime(
            df[col],
            errors="coerce",
            utc=True,
        )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove rows that cannot produce a valid target.
    df = df.dropna(subset=["created_at", "actual_delivery_time"])

    # Construct target.
    df["delivery_duration"] = (
        df["actual_delivery_time"] - df["created_at"]
    ).dt.total_seconds()

    # Invalid target values.
    df = df.loc[df["delivery_duration"] > 0].copy()

    # Features where negative values are invalid
    nonnegative_features = [
        "min_item_price",
        "total_onshift_dashers",
        "total_busy_dashers",
        "total_outstanding_orders",
    ]

    # Replace invalid negative values with NaN
    for col in nonnegative_features:
        df[col] = df[col].mask(df[col] < 0)

    # Missing categorical values.
    categorical_features = [
        "market_id",
        "order_protocol",
        "store_primary_category",
    ]

    for col in categorical_features:
        df[col] = df[col].fillna("unknown").astype(str)

    return df
