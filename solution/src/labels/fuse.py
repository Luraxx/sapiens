"""Combine multiple weak label sources into a fused pseudo ground-truth label."""

from __future__ import annotations

import numpy as np


def majority_vote(*label_arrays: np.ndarray, min_votes: int = 2) -> np.ndarray:
    """Fuse binary label arrays by majority vote.

    Args:
        *label_arrays: Variable number of (H, W) uint8 binary arrays.
        min_votes: Minimum number of sources that must agree for a positive label.

    Returns:
        (H, W) uint8 fused binary label.
    """
    stack = np.stack(label_arrays, axis=0)  # (N_sources, H, W)
    vote_count = stack.sum(axis=0)
    return (vote_count >= min_votes).astype(np.uint8)


def confidence_weighted_fusion(
    labels: list[np.ndarray],
    weights: list[float],
    threshold: float = 0.5,
) -> np.ndarray:
    """Fuse binary labels using confidence-based weights.

    Args:
        labels: List of (H, W) uint8 binary arrays.
        weights: Weight for each label source (higher = more trusted).
        threshold: Weighted sum threshold for positive label.

    Returns:
        (H, W) uint8 fused binary label.
    """
    weighted_sum = np.zeros_like(labels[0], dtype=np.float32)
    total_weight = sum(weights)

    for label, w in zip(labels, weights):
        weighted_sum += label.astype(np.float32) * w

    normalized = weighted_sum / total_weight
    return (normalized >= threshold).astype(np.uint8)


def union_fusion(*label_arrays: np.ndarray) -> np.ndarray:
    """Positive if ANY source flags deforestation (aggressive / high recall)."""
    stack = np.stack(label_arrays, axis=0)
    return (stack.max(axis=0) > 0).astype(np.uint8)


def intersection_fusion(*label_arrays: np.ndarray) -> np.ndarray:
    """Positive only if ALL sources agree (conservative / high precision)."""
    stack = np.stack(label_arrays, axis=0)
    return (stack.min(axis=0) > 0).astype(np.uint8)
