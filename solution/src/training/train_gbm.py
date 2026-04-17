"""Training loop for the LightGBM baseline model."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from solution.src.models.baseline_gbm import train_lightgbm, predict_lightgbm
from solution.src.training.cross_val import spatial_kfold

logger = logging.getLogger(__name__)


def train_gbm_pipeline(
    X: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    tile_ids: list[str],
    n_folds: int = 5,
    output_dir: Path | None = None,
) -> list[dict]:
    """Train LightGBM with spatial cross-validation.

    Args:
        X: dict mapping tile_id → (N_pixels, C) feature arrays.
        y: dict mapping tile_id → (N_pixels,) label arrays.
        tile_ids: list of tile IDs to use.
        n_folds: number of spatial folds.
        output_dir: Where to save fold models.

    Returns:
        List of fold results with metrics.
    """
    splits = spatial_kfold(tile_ids, n_folds)
    results = []

    for fold_idx, (train_ids, val_ids) in enumerate(splits):
        logger.info(f"Fold {fold_idx + 1}/{n_folds}: train={len(train_ids)} tiles, val={len(val_ids)} tiles")

        X_train = np.concatenate([X[tid] for tid in train_ids if tid in X])
        y_train = np.concatenate([y[tid] for tid in train_ids if tid in y])
        X_val = np.concatenate([X[tid] for tid in val_ids if tid in X])
        y_val = np.concatenate([y[tid] for tid in val_ids if tid in y])

        model = train_lightgbm(X_train, y_train, X_val, y_val)

        # Evaluate
        y_prob = predict_lightgbm(model, X_val)
        y_pred = (y_prob >= 0.5).astype(np.uint8)

        tp = ((y_pred == 1) & (y_val == 1)).sum()
        fp = ((y_pred == 1) & (y_val == 0)).sum()
        fn = ((y_pred == 0) & (y_val == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        fold_result = {
            "fold": fold_idx,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "model": model,
        }
        results.append(fold_result)
        logger.info(f"  F1={f1:.4f} | P={precision:.4f} | R={recall:.4f}")

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(output_dir / f"lgbm_fold{fold_idx}.txt"))

    return results
