"""Spatial cross-validation: split tiles into folds for geographic generalization."""

from __future__ import annotations

import numpy as np
import geopandas as gpd


def spatial_kfold(
    tile_ids: list[str],
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[list[str], list[str]]]:
    """Split tile IDs into k folds for spatial cross-validation.

    Returns:
        List of (train_tile_ids, val_tile_ids) tuples.
    """
    rng = np.random.default_rng(seed)
    ids = np.array(tile_ids)
    rng.shuffle(ids)

    folds = np.array_split(ids, n_folds)
    splits = []
    for i in range(n_folds):
        val_ids = folds[i].tolist()
        train_ids = [tid for j, fold in enumerate(folds) if j != i for tid in fold]
        splits.append((train_ids, val_ids))
    return splits
