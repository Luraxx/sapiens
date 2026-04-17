"""CRS reprojection utilities — align all rasters to a common grid."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform


def reproject_to_target(
    source_data: np.ndarray,
    src_transform,
    src_crs,
    dst_transform,
    dst_crs,
    dst_shape: tuple[int, int],
    resampling: Resampling = Resampling.nearest,
) -> np.ndarray:
    """Reproject a 2D array from source CRS/transform to destination CRS/transform.

    Args:
        source_data: 2D array (H, W).
        src_transform: Affine transform of the source.
        src_crs: CRS of the source.
        dst_transform: Affine transform of the destination grid.
        dst_crs: CRS of the destination grid.
        dst_shape: (H, W) of the destination grid.
        resampling: Resampling method.

    Returns:
        Reprojected 2D array of shape dst_shape.
    """
    destination = np.zeros(dst_shape, dtype=source_data.dtype)
    reproject(
        source=source_data,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
    )
    return destination


def reproject_multiband(
    source_data: np.ndarray,
    src_transform,
    src_crs,
    dst_transform,
    dst_crs,
    dst_shape: tuple[int, int],
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Reproject a multi-band array (C, H, W) band by band."""
    C = source_data.shape[0]
    result = np.zeros((C, *dst_shape), dtype=source_data.dtype)
    for i in range(C):
        result[i] = reproject_to_target(
            source_data[i], src_transform, src_crs,
            dst_transform, dst_crs, dst_shape, resampling,
        )
    return result
