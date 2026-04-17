"""Load Sentinel-1, Sentinel-2, AEF embeddings, and label rasters by tile ID."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import rasterio

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from solution.config import S1_DIR, S2_DIR, AEF_DIR, LABEL_DIR


# ── Sentinel-2 ─────────────────────────────────────────────────────────────

def load_s2(tile_id: str, year: int, month: int,
            split: Literal["train", "test"] = "train") -> tuple[np.ndarray, dict]:
    """Load a Sentinel-2 tile. Returns (bands, meta) where bands is (C, H, W)."""
    path = S2_DIR / split / f"{tile_id}__s2_l2a" / f"{tile_id}__s2_l2a_{year}_{month}.tif"
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        meta = {"transform": src.transform, "crs": src.crs, "shape": src.shape}
    return data, meta


# ── Sentinel-1 ─────────────────────────────────────────────────────────────

def load_s1(tile_id: str, year: int, month: int, orbit: str = "ascending",
            split: Literal["train", "test"] = "train") -> tuple[np.ndarray, dict]:
    """Load a Sentinel-1 tile. Returns (bands, meta) where bands is (1, H, W)."""
    path = S1_DIR / split / f"{tile_id}__s1_rtc" / f"{tile_id}__s1_rtc_{year}_{month}_{orbit}.tif"
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        meta = {"transform": src.transform, "crs": src.crs, "shape": src.shape}
    return data, meta


# ── AlphaEarth Foundation Embeddings ───────────────────────────────────────

def load_aef(tile_id: str, year: int,
             split: Literal["train", "test"] = "train") -> tuple[np.ndarray, dict]:
    """Load AEF embeddings. Returns (bands, meta) where bands is (64, H, W)."""
    path = AEF_DIR / split / f"{tile_id}_{year}.tiff"
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        meta = {"transform": src.transform, "crs": src.crs, "shape": src.shape}
    return data, meta


# ── Labels ─────────────────────────────────────────────────────────────────

def load_radd(tile_id: str) -> tuple[np.ndarray, dict]:
    """Load RADD label raster. Returns (data_2d, meta)."""
    path = LABEL_DIR / "radd" / f"radd_{tile_id}_labels.tif"
    with rasterio.open(path) as src:
        data = src.read(1)
        meta = {"transform": src.transform, "crs": src.crs, "shape": src.shape}
    return data, meta


def load_gladl(tile_id: str, year_suffix: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load GLAD-L alert + alertDate for a given year suffix (e.g. '24').
    Returns (alert, alert_date, meta)."""
    alert_path = LABEL_DIR / "gladl" / f"gladl_{tile_id}_alert{year_suffix}.tif"
    date_path  = LABEL_DIR / "gladl" / f"gladl_{tile_id}_alertDate{year_suffix}.tif"
    with rasterio.open(alert_path) as src:
        alert = src.read(1)
        meta = {"transform": src.transform, "crs": src.crs, "shape": src.shape}
    with rasterio.open(date_path) as src:
        alert_date = src.read(1)
    return alert, alert_date, meta


def load_glads2(tile_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load GLAD-S2 alert + alertDate.
    Returns (alert, alert_date, meta)."""
    alert_path = LABEL_DIR / "glads2" / f"glads2_{tile_id}_alert.tif"
    date_path  = LABEL_DIR / "glads2" / f"glads2_{tile_id}_alertDate.tif"
    with rasterio.open(alert_path) as src:
        alert = src.read(1)
        meta = {"transform": src.transform, "crs": src.crs, "shape": src.shape}
    with rasterio.open(date_path) as src:
        alert_date = src.read(1)
    return alert, alert_date, meta
