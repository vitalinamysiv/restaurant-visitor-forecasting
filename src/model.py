import os
import joblib

from catboost import CatBoostRegressor


def train_model(
    X_train,
    y_train
):

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        loss_function="MAE",
        random_seed=42,
        verbose=100
    )


    model.fit(
        X_train,
        y_train
    )


    return model



def save_model(
    model,
    path: str = "models/catboost_model.pkl"
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )


    joblib.dump(
        model,
        path
    )



def load_model(
    path: str = "models/catboost_model.pkl"
):

    return joblib.load(path)
