"""Decode raw label encodings from RADD, GLAD-L, and GLAD-S2 into binary masks."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np


# ── RADD ───────────────────────────────────────────────────────────────────

def decode_radd(
    raw: np.ndarray,
    min_confidence: int = 2,
    after_year: int = 2020,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode RADD encoded labels.

    Encoding: confidence * 10000 + days_since_epoch, epoch = 2018-12-31.
    confidence: 2=nominal, 3=high. 0 = no alert.

    Returns:
        binary: (H, W) uint8 — 1 = deforestation, 0 = no deforestation.
        dates:  (H, W) int32 — ordinal date of alert, 0 = no alert.
    """
    epoch = datetime(2018, 12, 31)
    cutoff_days = (datetime(after_year + 1, 1, 1) - epoch).days

    nonzero = raw > 0
    conf = np.where(nonzero, raw // 10000, 0).astype(np.int32)
    days = np.where(nonzero, raw % 10000, 0).astype(np.int32)

    valid = nonzero & (conf >= min_confidence) & (days >= cutoff_days)
    binary = valid.astype(np.uint8)

    ordinal_epoch = epoch.toordinal()
    dates = np.where(valid, ordinal_epoch + days, 0).astype(np.int32)

    return binary, dates


# ── GLAD-L ─────────────────────────────────────────────────────────────────

def decode_gladl(
    alert: np.ndarray,
    alert_date: np.ndarray,
    year: int,
    min_confidence: int = 2,
    after_year: int = 2020,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode GLAD-L alert + alertDate for a specific year.

    alert: 0=no loss, 2=probable, 3=confirmed
    alertDate: day-of-year within year 20YY, 0=no alert

    Returns:
        binary: (H, W) uint8
        dates:  (H, W) int32 ordinal dates
    """
    if year <= after_year:
        return np.zeros_like(alert, dtype=np.uint8), np.zeros_like(alert, dtype=np.int32)

    valid = (alert >= min_confidence) & (alert_date > 0)
    binary = valid.astype(np.uint8)

    # Convert day-of-year to ordinal date
    jan1 = datetime(year, 1, 1).toordinal()
    dates = np.where(valid, jan1 + alert_date.astype(int) - 1, 0).astype(np.int32)

    return binary, dates


# ── GLAD-S2 ────────────────────────────────────────────────────────────────

def decode_glads2(
    alert: np.ndarray,
    alert_date: np.ndarray,
    min_confidence: int = 2,
    after_year: int = 2020,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode GLAD-S2 alert + alertDate.

    alert: 0=no loss, 1=recent only, 2=low, 3=medium, 4=high confidence
    alertDate: days since 2019-01-01, 0=no alert

    Returns:
        binary: (H, W) uint8
        dates:  (H, W) int32 ordinal dates
    """
    epoch = datetime(2019, 1, 1)
    cutoff_days = (datetime(after_year + 1, 1, 1) - epoch).days

    valid = (alert >= min_confidence) & (alert_date >= cutoff_days)
    binary = valid.astype(np.uint8)

    ordinal_epoch = epoch.toordinal()
    dates = np.where(valid, ordinal_epoch + alert_date.astype(int), 0).astype(np.int32)

    return binary, dates
