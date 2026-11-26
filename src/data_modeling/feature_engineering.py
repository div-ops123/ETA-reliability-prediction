# src/data_modeling/feature_engineering.py

import pandas as pd


def split_chronologically(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    df = df.sort_values("created_at").copy()

    cutoff = df["created_at"].quantile(1 - test_size)

    train_df = df.loc[df["created_at"] < cutoff].copy()
    test_df = df.loc[df["created_at"] >= cutoff].copy()

    return train_df, test_df


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["month"] = df["created_at"].dt.month
    df["day_of_month"] = df["created_at"].dt.day
    df["day_of_week"] = df["created_at"].dt.dayofweek
    df["hour"] = df["created_at"].dt.hour

    return df
