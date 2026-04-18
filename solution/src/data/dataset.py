"""Dataset builders for tabular (pixel-level) and patch-based models."""

from __future__ import annotations

import numpy as np


def sample_pixels(
    features: np.ndarray,
    labels: np.ndarray,
    sample_rate: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified sampling: keep ALL positive pixels, subsample negatives.

    Args:
        features: (C, H, W) feature array.
        labels:   (H, W) binary label array.
        sample_rate: Fraction of *negative* pixels to sample.
        seed: Random seed.

    Returns:
        X: (N, C) feature matrix.
        y: (N,) label vector.
    """
    C, H, W = features.shape
    flat_feat = features.reshape(C, -1).T  # (N, C)
    flat_lab = labels.ravel()              # (N,)

    pos_idx = np.where(flat_lab == 1)[0]
    neg_idx = np.where(flat_lab == 0)[0]

    rng = np.random.default_rng(seed)
    n_neg_sample = max(1, int(len(neg_idx) * sample_rate))
    neg_chosen = rng.choice(neg_idx, size=n_neg_sample, replace=False)

    idx = np.concatenate([pos_idx, neg_chosen])
    rng.shuffle(idx)

    return flat_feat[idx], flat_lab[idx]


def extract_patches(
    features: np.ndarray,
    labels: np.ndarray,
    patch_size: int = 64,
    stride: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract non-overlapping (or strided) patches from a feature/label map.

    Args:
        features: (C, H, W)
        labels:   (H, W)
        patch_size: Size of square patches.
        stride: Step between patches. Defaults to patch_size (no overlap).

    Returns:
        X_patches: (N, C, patch_size, patch_size)
        y_patches: (N, patch_size, patch_size)
    """
    if stride is None:
        stride = patch_size

    C, H, W = features.shape
    x_patches, y_patches = [], []

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            x_patches.append(features[:, i:i + patch_size, j:j + patch_size])
            y_patches.append(labels[i:i + patch_size, j:j + patch_size])

    return np.stack(x_patches), np.stack(y_patches)
