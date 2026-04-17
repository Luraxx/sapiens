"""Convert binary prediction rasters to GeoJSON submission files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from submission_utils import raster_to_geojson


def generate_submission(
    prediction_dir: str | Path,
    output_dir: str | Path,
    min_area_ha: float = 0.5,
) -> Path:
    """Convert all prediction rasters in a directory to GeoJSON and merge.

    Args:
        prediction_dir: Directory containing binary GeoTIFF predictions.
        output_dir: Directory to write individual + merged GeoJSON files.
        min_area_ha: Minimum polygon area in hectares.

    Returns:
        Path to the merged submission GeoJSON.
    """
    prediction_dir = Path(prediction_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features = []

    for tif_path in sorted(prediction_dir.glob("*.tif")):
        tile_id = tif_path.stem
        geojson_path = output_dir / f"{tile_id}.geojson"

        try:
            geojson = raster_to_geojson(
                raster_path=tif_path,
                output_path=geojson_path,
                min_area_ha=min_area_ha,
            )
            all_features.extend(geojson["features"])
            logger.info(f"  {tile_id}: {len(geojson['features'])} polygons")
        except ValueError as e:
            logger.warning(f"  {tile_id}: skipped — {e}")

    # Merge all features into a single GeoJSON
    merged = {
        "type": "FeatureCollection",
        "features": all_features,
    }
    merged_path = output_dir / "submission.geojson"
    with open(merged_path, "w") as f:
        json.dump(merged, f)

    logger.info(f"Submission saved: {merged_path} ({len(all_features)} total polygons)")
    return merged_path
