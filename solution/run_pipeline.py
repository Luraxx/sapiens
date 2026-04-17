"""End-to-end pipeline: feature engineering → train → predict → submit.

Usage (from repo root):
    .venv/bin/python solution/run_pipeline.py
"""

import logging
import sys
import gc
import pickle
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solution.config import (
    DATA_DIR, S2_DIR, S1_DIR, AEF_DIR, LABEL_DIR,
    PRED_DIR, SUBMISSION_DIR, OUTPUT_DIR,
    DEFORESTATION_CUTOFF_YEAR, LABEL_FUSION_MIN_VOTES,
    PIXEL_SAMPLE_RATE, RANDOM_SEED, N_FOLDS,
    GLADS2_MIN_CONFIDENCE, RADD_CONFIDENCE_THRESHOLD,
)
from solution.src.data.inventory import list_tile_ids, list_s2_timesteps, list_s1_timesteps, list_aef_years
from solution.src.data.loader import load_s2, load_s1, load_radd, load_glads2, load_aef
from solution.src.data.reproject import reproject_to_target, reproject_multiband
from solution.src.labels.decode import decode_radd, decode_glads2
from solution.src.labels.fuse import majority_vote, confidence_weighted_fusion
from solution.src.features.indices import all_indices, ndvi, nbr, ndmi
from solution.src.features.temporal import temporal_stats, change_features, linear_trend
from solution.src.data.dataset import sample_pixels
from solution.src.models.baseline_gbm import train_lightgbm, predict_lightgbm
from solution.src.inference.predict import predict_tile_gbm, save_prediction_raster
from solution.src.inference.submit import generate_submission

from rasterio.warp import Resampling


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_features_for_tile(
    tile_id: str,
    split: str = "train",
) -> tuple[np.ndarray, dict] | None:
    """Build a feature stack (C, H, W) for one tile from all modalities.

    Returns (features, ref_meta) or None if data is insufficient.
    """
    s2_steps = list_s2_timesteps(tile_id, split)
    s1_steps = list_s1_timesteps(tile_id, split)

    # ── Reference grid: prefer S2, fallback to S1 ──────────────────
    if s2_steps:
        ref_data, ref_meta = load_s2(tile_id, s2_steps[0][0], s2_steps[0][1], split)
        ref_transform = ref_meta["transform"]
        ref_crs = ref_meta["crs"]
        ref_shape = ref_data.shape[1:]  # (H, W)
    elif s1_steps:
        ref_data, ref_meta = load_s1(tile_id, s1_steps[0][0], s1_steps[0][1], s1_steps[0][2], split)
        ref_transform = ref_meta["transform"]
        ref_crs = ref_meta["crs"]
        ref_shape = ref_data.shape[1:]  # (H, W)
        logger.info(f"  {tile_id}: using S1 as reference grid ({ref_shape}, {ref_crs})")
    else:
        logger.warning(f"  {tile_id}: no S2 or S1 data, skipping")
        return None

    feature_layers = []
    feature_names = []

    # ── S2: vegetation indices + raw bands over time ────────────────
    INDEX_NAMES = ["ndvi", "nbr", "ndmi", "evi"]
    RAW_BAND_NAMES = ["red", "nir", "swir1", "swir2"]

    if s2_steps:
        from solution.src.features.indices import _get_band_indices

        baseline_indices = {k: [] for k in INDEX_NAMES}
        post_indices = {k: [] for k in INDEX_NAMES}
        baseline_raw = {k: [] for k in RAW_BAND_NAMES}
        post_raw = {k: [] for k in RAW_BAND_NAMES}

        for year, month in s2_steps:
            try:
                s2_data, s2_meta = load_s2(tile_id, year, month, split)
            except Exception:
                continue
            # Reproject to reference grid if shape/CRS differs
            if s2_data.shape[1:] != ref_shape or s2_meta["crs"] != ref_crs:
                try:
                    s2_data = reproject_multiband(
                        s2_data, s2_meta["transform"], s2_meta["crs"],
                        ref_transform, ref_crs, ref_shape,
                        Resampling.bilinear,
                    )
                except Exception:
                    continue
            idx = all_indices(s2_data)
            bucket_idx = baseline_indices if year <= DEFORESTATION_CUTOFF_YEAR else post_indices
            for name in INDEX_NAMES:
                bucket_idx[name].append(idx[name])

            # Raw bands
            bi = _get_band_indices(s2_data.shape[0])
            raw_map = {"red": s2_data[bi["B04"]], "nir": s2_data[bi["B08"]],
                        "swir1": s2_data[bi["B11"]], "swir2": s2_data[bi["B12"]]}
            bucket_raw = baseline_raw if year <= DEFORESTATION_CUTOFF_YEAR else post_raw
            for bname, bval in raw_map.items():
                bucket_raw[bname].append(bval.astype(np.float32))

        # Temporal features for each index
        for idx_name in INDEX_NAMES:
            all_ts = baseline_indices[idx_name] + post_indices[idx_name]
            if len(all_ts) < 2:
                continue
            ts_arr = np.stack(all_ts, axis=0)  # (T, H, W)
            stats = temporal_stats(ts_arr)
            for stat_name, arr in stats.items():
                feature_layers.append(arr)
                feature_names.append(f"s2_{idx_name}_{stat_name}")

            # Trend
            feature_layers.append(linear_trend(ts_arr))
            feature_names.append(f"s2_{idx_name}_trend")

            # Change detection: baseline vs post
            if baseline_indices[idx_name] and post_indices[idx_name]:
                base_arr = np.stack(baseline_indices[idx_name], axis=0)
                post_arr = np.stack(post_indices[idx_name], axis=0)
                chg = change_features(base_arr, post_arr)
                for chg_name, arr in chg.items():
                    feature_layers.append(arr)
                    feature_names.append(f"s2_{idx_name}_{chg_name}")

        # Raw band temporal features (mean, std, change)
        for bname in RAW_BAND_NAMES:
            all_raw = baseline_raw[bname] + post_raw[bname]
            if len(all_raw) < 2:
                continue
            raw_ts = np.stack(all_raw, axis=0)
            raw_mean = np.nanmean(raw_ts, axis=0)
            raw_std = np.nanstd(raw_ts, axis=0)
            feature_layers.append(np.nan_to_num(raw_mean, nan=0.0))
            feature_names.append(f"s2_raw_{bname}_mean")
            feature_layers.append(np.nan_to_num(raw_std, nan=0.0))
            feature_names.append(f"s2_raw_{bname}_std")
            if baseline_raw[bname] and post_raw[bname]:
                base_mean = np.nanmean(np.stack(baseline_raw[bname]), axis=0)
                post_mean = np.nanmean(np.stack(post_raw[bname]), axis=0)
                feature_layers.append(np.nan_to_num(post_mean - base_mean, nan=0.0))
                feature_names.append(f"s2_raw_{bname}_change")

    # ── S1: backscatter temporal features ───────────────────────────
    if s1_steps:
        s1_baseline, s1_post = [], []
        for year, month, orbit in s1_steps:
            try:
                s1_data, s1_meta = load_s1(tile_id, year, month, orbit, split)
                vv = s1_data[0]
                # Convert to dB
                vv_db = np.where(vv > 0, 10 * np.log10(vv + 1e-10), np.nan)
                # Reproject to S2 grid if needed
                if s1_meta["crs"] != ref_crs or s1_meta["shape"] != ref_shape:
                    vv_db = reproject_to_target(
                        vv_db, s1_meta["transform"], s1_meta["crs"],
                        ref_transform, ref_crs, ref_shape,
                        Resampling.bilinear,
                    )
                bucket = s1_baseline if year <= DEFORESTATION_CUTOFF_YEAR else s1_post
                bucket.append(vv_db)
            except Exception:
                continue

        all_s1 = s1_baseline + s1_post
        if len(all_s1) >= 2:
            s1_ts = np.stack(all_s1, axis=0)
            s1_stats = temporal_stats(s1_ts)
            for stat_name, arr in s1_stats.items():
                feature_layers.append(np.nan_to_num(arr, nan=0.0))
                feature_names.append(f"s1_vv_{stat_name}")
            feature_layers.append(np.nan_to_num(linear_trend(s1_ts), nan=0.0))
            feature_names.append("s1_vv_trend")

            if s1_baseline and s1_post:
                chg = change_features(np.stack(s1_baseline), np.stack(s1_post))
                for chg_name, arr in chg.items():
                    feature_layers.append(np.nan_to_num(arr, nan=0.0))
                    feature_names.append(f"s1_vv_{chg_name}")

    # ── AEF: foundation model embeddings ────────────────────────────
    aef_years = list_aef_years(tile_id, split)
    if aef_years:
        baseline_year = min(aef_years)
        post_years = sorted([y for y in aef_years if y > DEFORESTATION_CUTOFF_YEAR])

        try:
            aef_base, aef_meta = load_aef(tile_id, baseline_year, split)
            # Reproject to S2 grid
            aef_base_reproj = reproject_multiband(
                aef_base, aef_meta["transform"], aef_meta["crs"],
                ref_transform, ref_crs, ref_shape,
                Resampling.bilinear,
            )
            n_aef_dims = aef_base_reproj.shape[0]  # 64

            # Full baseline embedding dims (64 features)
            for d in range(n_aef_dims):
                feature_layers.append(np.nan_to_num(aef_base_reproj[d], nan=0.0))
                feature_names.append(f"aef_base_d{d}")

            # Change vs latest post year
            if post_years:
                latest_year = max(post_years)
                aef_post, aef_post_meta = load_aef(tile_id, latest_year, split)
                aef_post_reproj = reproject_multiband(
                    aef_post, aef_post_meta["transform"], aef_post_meta["crs"],
                    ref_transform, ref_crs, ref_shape,
                    Resampling.bilinear,
                )

                # Full difference embedding (64 features)
                diff = aef_post_reproj - aef_base_reproj
                for d in range(n_aef_dims):
                    feature_layers.append(np.nan_to_num(diff[d], nan=0.0))
                    feature_names.append(f"aef_diff_d{d}")

                # Cosine distance and L2 distance (scalar summaries)
                dot = np.sum(aef_base_reproj * aef_post_reproj, axis=0)
                norm_a = np.linalg.norm(aef_base_reproj, axis=0) + 1e-8
                norm_b = np.linalg.norm(aef_post_reproj, axis=0) + 1e-8
                cos_dist = 1.0 - dot / (norm_a * norm_b)
                l2_dist = np.linalg.norm(aef_post_reproj - aef_base_reproj, axis=0)

                feature_layers.append(np.nan_to_num(cos_dist, nan=0.0))
                feature_names.append("aef_cosine_dist")
                feature_layers.append(np.nan_to_num(l2_dist, nan=0.0))
                feature_names.append("aef_l2_dist")

            # Max year-over-year AEF change (captures abrupt changes)
            if len(aef_years) >= 2:
                aef_all_reproj = [aef_base_reproj]
                for yr in sorted(aef_years)[1:]:
                    try:
                        aef_yr, aef_yr_meta = load_aef(tile_id, yr, split)
                        aef_yr_reproj = reproject_multiband(
                            aef_yr, aef_yr_meta["transform"], aef_yr_meta["crs"],
                            ref_transform, ref_crs, ref_shape,
                            Resampling.bilinear,
                        )
                        aef_all_reproj.append(aef_yr_reproj)
                    except Exception:
                        continue

                if len(aef_all_reproj) >= 2:
                    yoy_l2 = []
                    for j in range(1, len(aef_all_reproj)):
                        yoy = np.linalg.norm(aef_all_reproj[j] - aef_all_reproj[j-1], axis=0)
                        yoy_l2.append(yoy)
                    yoy_stack = np.stack(yoy_l2, axis=0)
                    feature_layers.append(np.nan_to_num(np.nanmax(yoy_stack, axis=0), nan=0.0))
                    feature_names.append("aef_max_yoy_l2")
                    feature_layers.append(np.nan_to_num(np.nanstd(yoy_stack, axis=0), nan=0.0))
                    feature_names.append("aef_yoy_l2_std")
        except Exception as e:
            logger.warning(f"  {tile_id}: AEF error: {e}")

    if not feature_layers:
        logger.warning(f"  {tile_id}: no features built")
        return None

    # Ensure canonical feature order — always produce the same number of features
    CANONICAL_FEATURES = []
    for idx_name in ["ndvi", "nbr", "ndmi", "evi"]:
        for stat_name in ["mean", "std", "min", "max", "median", "range"]:
            CANONICAL_FEATURES.append(f"s2_{idx_name}_{stat_name}")
        CANONICAL_FEATURES.append(f"s2_{idx_name}_trend")
        for chg_name in ["mean_diff", "delta_min", "delta_max", "max_decrease"]:
            CANONICAL_FEATURES.append(f"s2_{idx_name}_{chg_name}")
    for bname in ["red", "nir", "swir1", "swir2"]:
        CANONICAL_FEATURES.append(f"s2_raw_{bname}_mean")
        CANONICAL_FEATURES.append(f"s2_raw_{bname}_std")
        CANONICAL_FEATURES.append(f"s2_raw_{bname}_change")
    for stat_name in ["mean", "std", "min", "max", "median", "range"]:
        CANONICAL_FEATURES.append(f"s1_vv_{stat_name}")
    CANONICAL_FEATURES.append("s1_vv_trend")
    for chg_name in ["mean_diff", "delta_min", "delta_max", "max_decrease"]:
        CANONICAL_FEATURES.append(f"s1_vv_{chg_name}")
    for d in range(64):
        CANONICAL_FEATURES.append(f"aef_base_d{d}")
    for d in range(64):
        CANONICAL_FEATURES.append(f"aef_diff_d{d}")
    CANONICAL_FEATURES.append("aef_cosine_dist")
    CANONICAL_FEATURES.append("aef_l2_dist")
    CANONICAL_FEATURES.append("aef_max_yoy_l2")
    CANONICAL_FEATURES.append("aef_yoy_l2_std")

    name_to_layer = dict(zip(feature_names, feature_layers))
    ordered_layers = []
    for fname in CANONICAL_FEATURES:
        if fname in name_to_layer:
            ordered_layers.append(name_to_layer[fname])
        else:
            ordered_layers.append(np.zeros(ref_shape, dtype=np.float32))

    features = np.stack(ordered_layers, axis=0).astype(np.float32)  # (C, H, W)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info(f"  {tile_id}: {features.shape[0]} features, shape {features.shape[1:]}")
    return features, ref_meta


def build_labels_for_tile(tile_id: str, ref_meta: dict) -> np.ndarray | None:
    """Build fused binary deforestation label for a training tile.

    Returns (H, W) uint8 binary label aligned to the ref_meta grid.
    """
    ref_transform = ref_meta["transform"]
    ref_crs = ref_meta["crs"]
    ref_shape = ref_meta["shape"]

    label_sources = []

    # ── RADD ──────────────────────────────────────────────────────
    try:
        radd_raw, radd_meta = load_radd(tile_id)
        radd_binary, _ = decode_radd(radd_raw, min_confidence=RADD_CONFIDENCE_THRESHOLD, after_year=DEFORESTATION_CUTOFF_YEAR)
        if radd_meta["crs"] != ref_crs or radd_meta["shape"] != ref_shape:
            radd_binary = reproject_to_target(
                radd_binary, radd_meta["transform"], radd_meta["crs"],
                ref_transform, ref_crs, ref_shape, Resampling.nearest,
            )
        label_sources.append(radd_binary)
    except Exception as e:
        logger.debug(f"  {tile_id}: RADD unavailable: {e}")

    # ── GLAD-S2 ──────────────────────────────────────────────────
    try:
        gs2_alert, gs2_date, gs2_meta = load_glads2(tile_id)
        gs2_binary, _ = decode_glads2(gs2_alert, gs2_date, min_confidence=GLADS2_MIN_CONFIDENCE, after_year=DEFORESTATION_CUTOFF_YEAR)
        if gs2_meta["crs"] != ref_crs or gs2_meta["shape"] != ref_shape:
            gs2_binary = reproject_to_target(
                gs2_binary, gs2_meta["transform"], gs2_meta["crs"],
                ref_transform, ref_crs, ref_shape, Resampling.nearest,
            )
        label_sources.append(gs2_binary)
    except Exception as e:
        logger.debug(f"  {tile_id}: GLAD-S2 unavailable: {e}")

    # ── GLAD-L (try multiple year suffixes) ──────────────────────
    from solution.src.data.loader import load_gladl
    from solution.src.labels.decode import decode_gladl
    gladl_combined = None
    for yy in range(21, 26):  # 2021-2025
        try:
            gl_alert, gl_date, gl_meta = load_gladl(tile_id, str(yy))
            gl_binary, _ = decode_gladl(gl_alert, gl_date, 2000 + yy, min_confidence=2, after_year=DEFORESTATION_CUTOFF_YEAR)
            if gl_meta["crs"] != ref_crs or gl_meta["shape"] != ref_shape:
                gl_binary = reproject_to_target(
                    gl_binary, gl_meta["transform"], gl_meta["crs"],
                    ref_transform, ref_crs, ref_shape, Resampling.nearest,
                )
            if gladl_combined is None:
                gladl_combined = gl_binary.copy()
            else:
                gladl_combined = np.maximum(gladl_combined, gl_binary)
        except Exception:
            continue
    if gladl_combined is not None:
        label_sources.append(gladl_combined)

    if not label_sources:
        logger.warning(f"  {tile_id}: no label sources available")
        return None

    if len(label_sources) == 1:
        return label_sources[0]

    # Fuse: use majority vote if 3 sources, else just agreement of 2
    min_v = min(LABEL_FUSION_MIN_VOTES, len(label_sources))
    fused = majority_vote(*label_sources, min_votes=min_v)
    pos = fused.sum()
    total = fused.size
    logger.info(f"  {tile_id}: fused label — {pos}/{total} positive ({100*pos/total:.2f}%)")
    return fused


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("  Deforestation Detection Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Inventory ──────────────────────────────────────────
    train_tiles = list_tile_ids("train")
    test_tiles = list_tile_ids("test")
    logger.info(f"Train tiles: {len(train_tiles)} | Test tiles: {len(test_tiles)}")

    if not train_tiles:
        logger.error("No training tiles found. Run `make download_data_from_s3` first.")
        sys.exit(1)

    model_path = OUTPUT_DIR / "lgbm_model.pkl"
    feature_names_path = OUTPUT_DIR / "feature_names.txt"

    # ── Step 2: Feature engineering + label fusion ─────────────────
    logger.info("\n── Step 2: Feature Engineering & Label Fusion ──")

    all_X = []
    all_y = []

    for i, tile_id in enumerate(train_tiles):
        logger.info(f"[{i+1}/{len(train_tiles)}] Processing {tile_id}")

        result = build_features_for_tile(tile_id, "train")
        if result is None:
            continue
        features, ref_meta = result

        labels = build_labels_for_tile(tile_id, ref_meta)
        if labels is None:
            continue

        # Sample pixels for tabular model
        X_tile, y_tile = sample_pixels(features, labels, sample_rate=PIXEL_SAMPLE_RATE, seed=RANDOM_SEED + i)
        all_X.append(X_tile)
        all_y.append(y_tile)
        logger.info(f"  Sampled {X_tile.shape[0]} pixels ({y_tile.sum()} positive)")

        # Free memory
        del features, labels, result
        gc.collect()

    if not all_X:
        logger.error("No training data could be built. Check data downloads.")
        sys.exit(1)

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    logger.info(f"\nTotal training pixels: {X_all.shape[0]} ({X_all.shape[1]} features)")
    logger.info(f"Positive rate: {y_all.mean():.4f}")

    # ── Step 3: Train LightGBM ────────────────────────────────────
    logger.info("\n── Step 3: Training LightGBM ──")

    # Simple train/val split (80/20 random)
    rng = np.random.default_rng(RANDOM_SEED)
    n = X_all.shape[0]
    idx = rng.permutation(n)
    split_at = int(0.8 * n)
    X_train, X_val = X_all[idx[:split_at]], X_all[idx[split_at:]]
    y_train, y_val = y_all[idx[:split_at]], y_all[idx[split_at:]]

    logger.info(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")

    model = train_lightgbm(X_train, y_train, X_val, y_val)

    # Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")

    # Validation metrics
    y_prob = predict_lightgbm(model, X_val)
    y_pred = (y_prob >= 0.5).astype(np.uint8)
    tp = ((y_pred == 1) & (y_val == 1)).sum()
    fp = ((y_pred == 1) & (y_val == 0)).sum()
    fn = ((y_pred == 0) & (y_val == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    logger.info(f"Validation: F1={f1:.4f} | Precision={precision:.4f} | Recall={recall:.4f}")

    del X_all, y_all, X_train, y_train, X_val, y_val
    gc.collect()

    # ── Step 4: Predict on test tiles ─────────────────────────────
    logger.info("\n── Step 4: Inference on Test Tiles ──")
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    for i, tile_id in enumerate(test_tiles):
        logger.info(f"[{i+1}/{len(test_tiles)}] Predicting {tile_id}")
        result = build_features_for_tile(tile_id, "test")
        if result is None:
            logger.warning(f"  {tile_id}: could not build features, skipping")
            continue
        features, ref_meta = result

        pred = predict_tile_gbm(model, features, threshold=0.45)

        # Morphological post-processing: close small gaps, remove tiny isolated pixels
        from scipy.ndimage import binary_closing, binary_opening, label as ndlabel
        struct = np.ones((3, 3), dtype=bool)
        pred = binary_closing(pred, structure=struct, iterations=1).astype(np.uint8)
        pred = binary_opening(pred, structure=struct, iterations=1).astype(np.uint8)
        # Remove connected components smaller than 6 pixels (~0.5 ha at 30m)
        labeled, n_labels = ndlabel(pred)
        for comp in range(1, n_labels + 1):
            if np.sum(labeled == comp) < 6:
                pred[labeled == comp] = 0

        # Find reference raster for CRS/transform
        s2_steps = list_s2_timesteps(tile_id, "test")
        s1_steps_t = list_s1_timesteps(tile_id, "test")
        if s2_steps:
            ref_path = S2_DIR / "test" / f"{tile_id}__s2_l2a" / f"{tile_id}__s2_l2a_{s2_steps[0][0]}_{s2_steps[0][1]}.tif"
        elif s1_steps_t:
            yr, mo, orb = s1_steps_t[0]
            ref_path = S1_DIR / "test" / f"{tile_id}__s1_rtc" / f"{tile_id}__s1_rtc_{yr}_{mo}_{orb}.tif"
        else:
            logger.warning(f"  {tile_id}: no reference raster found")
            continue

        out_path = PRED_DIR / f"pred_{tile_id}.tif"
        save_prediction_raster(pred, ref_path, out_path)
        logger.info(f"  {tile_id}: {pred.sum()} deforestation pixels ({100*pred.mean():.2f}%)")

        del features, pred, result
        gc.collect()

    # ── Step 5: Generate submission ───────────────────────────────
    logger.info("\n── Step 5: Generating Submission ──")
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    submission_path = generate_submission(PRED_DIR, SUBMISSION_DIR, min_area_ha=0.5)
    logger.info(f"\nSubmission file: {submission_path}")

    logger.info("=" * 60)
    logger.info("  Pipeline complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
