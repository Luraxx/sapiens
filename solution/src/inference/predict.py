"""Run inference on test tiles and produce binary prediction rasters."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio

logger = logging.getLogger(__name__)


def predict_tile_gbm(
    model,
    features: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run LightGBM prediction on a full tile's features.

    Args:
        model: Trained LightGBM Booster.
        features: (C, H, W) feature stack for one tile.
        threshold: Probability threshold for binary prediction.

    Returns:
        (H, W) uint8 binary prediction.
    """
    C, H, W = features.shape
    X = features.reshape(C, -1).T  # (N, C)
    probs = model.predict(X, num_iteration=model.best_iteration)
    binary = (probs >= threshold).astype(np.uint8)
    return binary.reshape(H, W)


def predict_tile_unet(
    model,
    features: np.ndarray,
    patch_size: int = 64,
    device: str = "cpu",
    threshold: float = 0.5,
) -> np.ndarray:
    """Run U-Net prediction on a full tile using sliding window.

    Args:
        model: Trained UNet model.
        features: (C, H, W) feature stack.
        patch_size: Patch size for inference.
        device: Torch device.
        threshold: Sigmoid threshold.

    Returns:
        (H, W) uint8 binary prediction.
    """
    import torch

    model.eval()
    C, H, W = features.shape
    pred = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                # Handle edge patches
                i_end = min(i + patch_size, H)
                j_end = min(j + patch_size, W)
                i_start = max(0, i_end - patch_size)
                j_start = max(0, j_end - patch_size)

                patch = features[:, i_start:i_end, j_start:j_end]
                x = torch.from_numpy(patch[None]).float().to(device)
                logits = model(x)[0, 0].cpu().numpy()
                prob = 1 / (1 + np.exp(-logits))  # sigmoid

                pred[i_start:i_end, j_start:j_end] += prob
                count[i_start:i_end, j_start:j_end] += 1.0

    pred = pred / np.maximum(count, 1.0)
    return (pred >= threshold).astype(np.uint8)


def save_prediction_raster(
    prediction: np.ndarray,
    reference_path: str | Path,
    output_path: str | Path,
) -> None:
    """Save a binary prediction as a GeoTIFF using a reference raster for CRS/transform.

    Args:
        prediction: (H, W) uint8 binary array.
        reference_path: Path to a raster with the target CRS and transform.
        output_path: Where to write the output GeoTIFF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_path) as ref:
        meta = ref.meta.copy()

    meta.update(dtype="uint8", count=1, nodata=0)

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(prediction, 1)

    logger.info(f"Saved prediction: {output_path}")
