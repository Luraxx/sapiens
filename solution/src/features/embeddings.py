"""Process AlphaEarth Foundation embeddings for feature engineering."""

from __future__ import annotations

import numpy as np

from solution.src.data.reproject import reproject_multiband


def embedding_change(
    emb_baseline: np.ndarray,
    emb_post: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute change features from annual AEF embeddings.

    Args:
        emb_baseline: (64, H, W) embedding for baseline year (2020).
        emb_post:     (64, H, W) embedding for a post year.

    Returns:
        Dict with delta embedding, cosine distance, L2 distance.
    """
    delta = emb_post - emb_baseline

    # Cosine distance per pixel
    dot = np.sum(emb_baseline * emb_post, axis=0)
    norm_a = np.linalg.norm(emb_baseline, axis=0) + 1e-8
    norm_b = np.linalg.norm(emb_post, axis=0) + 1e-8
    cosine_sim = dot / (norm_a * norm_b)

    # L2 distance per pixel
    l2_dist = np.linalg.norm(delta, axis=0)

    return {
        "delta_embedding": delta,          # (64, H, W)
        "cosine_distance": 1 - cosine_sim, # (H, W)
        "l2_distance": l2_dist,            # (H, W)
    }


def reduce_embedding(emb: np.ndarray, method: str = "pca", n_components: int = 8) -> np.ndarray:
    """Reduce 64-dim embedding to fewer dimensions per pixel.

    Args:
        emb: (64, H, W) embedding array.
        method: 'pca' or 'mean_pool'.
        n_components: Number of output components (for PCA).

    Returns:
        (n_components, H, W) or (1, H, W) reduced embedding.
    """
    C, H, W = emb.shape
    if method == "mean_pool":
        # Pool groups of channels
        group_size = C // n_components
        pooled = []
        for i in range(n_components):
            start = i * group_size
            end = start + group_size if i < n_components - 1 else C
            pooled.append(np.nanmean(emb[start:end], axis=0))
        return np.stack(pooled)

    elif method == "pca":
        from sklearn.decomposition import PCA
        flat = emb.reshape(C, -1).T  # (N_pixels, 64)
        valid_mask = np.isfinite(flat).all(axis=1)
        pca = PCA(n_components=n_components)
        result = np.zeros((flat.shape[0], n_components), dtype=np.float32)
        if valid_mask.sum() > n_components:
            result[valid_mask] = pca.fit_transform(flat[valid_mask])
        return result.T.reshape(n_components, H, W)

    raise ValueError(f"Unknown method: {method}")
