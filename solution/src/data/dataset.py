"""Dataset builders for tabular (pixel-level) and patch-based models."""

from __future__ import annotations

import numpy as np


def sample_pixels(
    features: np.ndarray,
    labels: np.ndarray,
    sample_rate: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly sample pixels from a feature map for tabular training.

    Args:
        features: (C, H, W) feature array.
        labels:   (H, W) binary label array.
        sample_rate: Fraction of pixels to sample.
        seed: Random seed.

    Returns:
        X: (N, C) feature matrix.
        y: (N,) label vector.
    """
    C, H, W = features.shape
    n_pixels = H * W
    n_sample = max(1, int(n_pixels * sample_rate))

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_pixels, size=n_sample, replace=False)

    X = features.reshape(C, -1).T[idx]  # (N, C)
    y = labels.ravel()[idx]              # (N,)
    return X, y


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
