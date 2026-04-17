"""Scan the data directory and build an inventory of available tiles and time steps."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import geopandas as gpd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from solution.config import S1_DIR, S2_DIR, AEF_DIR, LABEL_DIR, META_DIR


def list_tile_ids(split: Literal["train", "test"] = "train") -> list[str]:
    """Return sorted list of tile IDs from the metadata GeoJSON."""
    gdf = gpd.read_file(META_DIR / f"{split}_tiles.geojson")
    # Try common column names for tile ID
    for col in ("tile_id", "id", "name", "TILE_ID"):
        if col in gdf.columns:
            return sorted(gdf[col].tolist())
    # Fallback: first non-geometry column
    non_geom = [c for c in gdf.columns if c != "geometry"]
    return sorted(gdf[non_geom[0]].tolist()) if non_geom else []


def list_s2_timesteps(tile_id: str, split: Literal["train", "test"] = "train") -> list[tuple[int, int]]:
    """Return sorted list of (year, month) tuples available for a Sentinel-2 tile."""
    folder = S2_DIR / split / f"{tile_id}__s2_l2a"
    if not folder.exists():
        return []
    pattern = re.compile(rf"{re.escape(tile_id)}__s2_l2a_(\d{{4}})_(\d{{1,2}})\.tif")
    steps = []
    for p in folder.iterdir():
        m = pattern.match(p.name)
        if m:
            steps.append((int(m.group(1)), int(m.group(2))))
    return sorted(steps)


def list_s1_timesteps(tile_id: str, split: Literal["train", "test"] = "train") -> list[tuple[int, int, str]]:
    """Return sorted list of (year, month, orbit) tuples for a Sentinel-1 tile."""
    folder = S1_DIR / split / f"{tile_id}__s1_rtc"
    if not folder.exists():
        return []
    pattern = re.compile(rf"{re.escape(tile_id)}__s1_rtc_(\d{{4}})_(\d{{1,2}})_(ascending|descending)\.tif")
    steps = []
    for p in folder.iterdir():
        m = pattern.match(p.name)
        if m:
            steps.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    return sorted(steps)


def list_aef_years(tile_id: str, split: Literal["train", "test"] = "train") -> list[int]:
    """Return sorted list of years available for AlphaEarth embeddings."""
    folder = AEF_DIR / split
    if not folder.exists():
        return []
    pattern = re.compile(rf"{re.escape(tile_id)}_(\d{{4}})\.tiff?")
    years = []
    for p in folder.iterdir():
        m = pattern.match(p.name)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def tile_summary(tile_id: str, split: Literal["train", "test"] = "train") -> dict:
    """Return a summary dict of data availability for a tile."""
    return {
        "tile_id": tile_id,
        "split": split,
        "s2_timesteps": len(list_s2_timesteps(tile_id, split)),
        "s1_timesteps": len(list_s1_timesteps(tile_id, split)),
        "aef_years": list_aef_years(tile_id, split),
    }
