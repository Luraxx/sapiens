"""LightGBM / XGBoost pixel-level gradient boosting classifier."""

from __future__ import annotations

import numpy as np


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    params: dict | None = None,
):
    """Train a LightGBM binary classifier.

    Args:
        X_train: (N, C) feature matrix.
        y_train: (N,) binary labels.
        X_val: Optional validation features.
        y_val: Optional validation labels.
        params: LightGBM parameters override.

    Returns:
        Trained LightGBM Booster.
    """
    import lightgbm as lgb

    default_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 255,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "is_unbalance": True,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }
    if params:
        default_params.update(params)

    dtrain = lgb.Dataset(X_train, label=y_train)
    valid_sets = [dtrain]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(100)]

    if X_val is not None and y_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        valid_sets.append(dval)
        valid_names.append("val")
        callbacks.append(lgb.early_stopping(50))

    model = lgb.train(
        default_params,
        dtrain,
        num_boost_round=3000,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return model


def predict_lightgbm(model, X: np.ndarray) -> np.ndarray:
    """Predict probabilities using a trained LightGBM model.

    Returns: (N,) array of deforestation probabilities.
    """
    return model.predict(X, num_iteration=model.best_iteration)
