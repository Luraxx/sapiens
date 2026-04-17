"""Compute vegetation and spectral indices from Sentinel-2 bands."""

from __future__ import annotations

import numpy as np

# Band indices (0-based) within a Sentinel-2 multi-band array (C, H, W)
# Likely order (12 bands, B10/cirrus excluded from L2A):
# B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
# NOTE: Will be verified once data is available. If file has 13 bands,
#       shift B11 and B12 indices by +1.
_B02 = 1   # Blue
_B03 = 2   # Green
_B04 = 3   # Red
_B08 = 7   # NIR
_B8A = 8   # Narrow NIR
_B11 = 10  # SWIR1 (index 10 if B10 excluded, 11 if B10 present)
_B12 = 11  # SWIR2 (index 11 if B10 excluded, 12 if B10 present)


def _get_band_indices(n_bands: int) -> dict[str, int]:
    """Return band name → 0-based index mapping based on actual band count."""
    if n_bands == 13:
        # All bands present: B01-B12 including B8A and B10
        return {"B02": 1, "B03": 2, "B04": 3, "B08": 7, "B8A": 8, "B11": 11, "B12": 12}
    elif n_bands == 12:
        # B10 (cirrus) excluded — common for L2A
        return {"B02": 1, "B03": 2, "B04": 3, "B08": 7, "B8A": 8, "B11": 10, "B12": 11}
    else:
        # Fallback — assume same as 12 band
        return {"B02": 1, "B03": 2, "B04": 3, "B08": 7, "B8A": 8, "B11": min(10, n_bands-2), "B12": min(11, n_bands-1)}


def _safe_ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (a - b) / (a + b + eps)


def ndvi(s2: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index. (NIR - Red) / (NIR + Red)"""
    b = _get_band_indices(s2.shape[0])
    return _safe_ratio(s2[b["B08"]], s2[b["B04"]])


def evi(s2: np.ndarray) -> np.ndarray:
    """Enhanced Vegetation Index. 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)"""
    b = _get_band_indices(s2.shape[0])
    nir, red, blue = s2[b["B08"]], s2[b["B04"]], s2[b["B02"]]
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    return 2.5 * (nir - red) / (denom + 1e-6)


def nbr(s2: np.ndarray) -> np.ndarray:
    """Normalized Burn Ratio. (NIR - SWIR2) / (NIR + SWIR2)"""
    b = _get_band_indices(s2.shape[0])
    return _safe_ratio(s2[b["B08"]], s2[b["B12"]])


def ndmi(s2: np.ndarray) -> np.ndarray:
    """Normalized Difference Moisture Index. (NIR - SWIR1) / (NIR + SWIR1)"""
    b = _get_band_indices(s2.shape[0])
    return _safe_ratio(s2[b["B08"]], s2[b["B11"]])


def all_indices(s2: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all indices at once. Returns dict of name → (H, W) array."""
    return {
        "ndvi": ndvi(s2),
        "evi": evi(s2),
        "nbr": nbr(s2),
        "ndmi": ndmi(s2),
    }
