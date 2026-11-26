from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


onehot_cols = [
    "order_protocol",
    "market_id",
]

numeric_features = [
    # Order features
    "total_items",
    "subtotal",
    "num_distinct_items",
    "min_item_price",
    "max_item_price",
    # Marketplace features
    "total_onshift_dashers",
    "total_busy_dashers",
    "total_outstanding_orders",
    # Estimated durations
    "estimated_order_place_duration",
    "estimated_store_to_consumer_driving_duration",
    # Date features
    "month",
    "day_of_month",
    "day_of_week",
    "hour",
]


# Categorical preprocessing
categorical_preprocessor = OneHotEncoder(
    handle_unknown="ignore", sparse_output=False
)


# Numerical preprocessing
numerical_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler())
    ]
)


# Full preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_preprocessor, onehot_cols),
        ("numerical", numerical_preprocessor, numeric_features),
    ],
    remainder="drop",
)
