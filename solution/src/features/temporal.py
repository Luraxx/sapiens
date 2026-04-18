"""Temporal feature aggregation: statistics, trends, and change detection."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def temporal_stats(time_series: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-pixel temporal statistics from a time series stack.

    Args:
        time_series: (T, H, W) array of a single feature over T time steps.

    Returns:
        Dict of statistic name → (H, W) array.
    """
    # Single sort for all quantile-based stats (min, p10, p25, median, p75, p90, max)
    _pcts = np.nanpercentile(time_series, [0, 10, 25, 50, 75, 90, 100], axis=0)
    _min, _p10, _p25, _median, _p75, _p90, _max = _pcts
    return {
        "mean":   np.nanmean(time_series, axis=0),
        "std":    np.nanstd(time_series, axis=0),
        "min":    _min,
        "max":    _max,
        "median": _median,
        "range":  _max - _min,
        "p10":    _p10,
        "p25":    _p25,
        "p75":    _p75,
        "p90":    _p90,
    }


def linear_trend(time_series: np.ndarray) -> np.ndarray:
    """Compute per-pixel linear trend (slope) over time.

    Args:
        time_series: (T, H, W) array.

    Returns:
        (H, W) array of slope values.
    """
    T, H, W = time_series.shape
    x = np.arange(T, dtype=np.float32)
    flat = time_series.reshape(T, -1)  # (T, N)

    # Vectorized linear regression: slope = cov(x, y) / var(x)
    x_mean = x.mean()
    y_mean = np.nanmean(flat, axis=0)
    cov = np.nanmean((x[:, None] - x_mean) * (flat - y_mean[None, :]), axis=0)
    var_x = np.var(x)
    slope = cov / (var_x + 1e-10)

    return slope.reshape(H, W)


def change_features(
    baseline: np.ndarray,
    post: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute change features between baseline period and post period.

    Args:
        baseline: (T1, H, W) time series for baseline period (e.g. 2020).
        post:     (T2, H, W) time series for post period (e.g. 2021+).

    Returns:
        Dict of change feature name → (H, W) array.
    """
    base_mean = np.nanmean(baseline, axis=0)
    post_mean = np.nanmean(post, axis=0)

    return {
        "delta_mean": post_mean - base_mean,
        "delta_ratio": (post_mean - base_mean) / (base_mean + 1e-6),
        "delta_min": np.nanmin(post, axis=0) - np.nanmin(baseline, axis=0),
        "delta_max": np.nanmax(post, axis=0) - np.nanmax(baseline, axis=0),
    }
