import os
from typing import Any

import joblib
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Train Ridge regression inside a scaling pipeline.

    Scaling is fitted on the training set only,
    then applied to the validation set.
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: list[str] | None = None
) -> CatBoostRegressor:
    """
    Train CatBoost regressor with fixed random seed.

    `cat_features` — list of categorical feature names
    (e.g. ['restaurant_id']) if you decide to include them.
    """
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        loss_function="MAE",
        random_seed=42,
        verbose=100,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features or [],
    )
    return model


def save_model(
    model: Any,
    path: str = "models/catboost_model.pkl"
) -> None:
    """Save model to disk, creating the directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str = "models/catboost_model.pkl") -> Any:
    """Load model from disk, raising a clear error if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Файл модели не найден: {path}. "
            f"Сначала обучите модель (см. README)."
        )
    return joblib.load(path)
