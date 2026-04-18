"""Data & model analysis script — identify optimization opportunities."""

import sys, pickle, gc
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solution.config import *
from solution.src.data.inventory import list_tile_ids, list_s2_timesteps, list_s1_timesteps, list_aef_years
from solution.src.data.loader import load_s2, load_radd, load_glads2, load_aef, load_gladl
from solution.src.labels.decode import decode_radd, decode_glads2, decode_gladl
from solution.src.labels.fuse import majority_vote
from solution.src.data.reproject import reproject_to_target
from solution.src.features.indices import all_indices
from rasterio.warp import Resampling

OUTPUT = OUTPUT_DIR / "analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 1) Feature importance from saved model
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("1) FEATURE IMPORTANCE")
print("=" * 60)

model_path = OUTPUT_DIR / "lgbm_model.pkl"
if model_path.exists():
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    imp = model.feature_importance(importance_type="gain")
    
    # Build canonical feature names
    _STAT_NAMES = ["mean", "std", "min", "max", "median", "range", "p10", "p25", "p75", "p90"]
    _CHG_NAMES = ["delta_mean", "delta_ratio", "delta_min", "delta_max"]
    FEATURE_NAMES = []
    for idx_name in ["ndvi", "nbr", "ndmi", "evi"]:
        for s in _STAT_NAMES: FEATURE_NAMES.append(f"s2_{idx_name}_{s}")
        FEATURE_NAMES.append(f"s2_{idx_name}_trend")
        for c in _CHG_NAMES: FEATURE_NAMES.append(f"s2_{idx_name}_{c}")
    for bname in ["red", "nir", "swir1", "swir2"]:
        for s in _STAT_NAMES: FEATURE_NAMES.append(f"s2_raw_{bname}_{s}")
        FEATURE_NAMES.append(f"s2_raw_{bname}_trend")
        for c in _CHG_NAMES: FEATURE_NAMES.append(f"s2_raw_{bname}_{c}")
    for s in _STAT_NAMES: FEATURE_NAMES.append(f"s1_vv_{s}")
    FEATURE_NAMES.append("s1_vv_trend")
    for c in _CHG_NAMES: FEATURE_NAMES.append(f"s1_vv_{c}")
    for d in range(64): FEATURE_NAMES.append(f"aef_base_d{d}")
    for d in range(64): FEATURE_NAMES.append(f"aef_diff_d{d}")
    FEATURE_NAMES += ["aef_cosine_dist", "aef_l2_dist", "aef_max_yoy_l2", "aef_yoy_l2_std"]
    
    pairs = sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])
    
    print("\nTop 40 features by gain:")
    for name, g in pairs[:40]:
        print(f"  {g:>12.1f}  {name}")
    
    print(f"\nBottom 20 features (near zero importance):")
    for name, g in pairs[-20:]:
        print(f"  {g:>12.1f}  {name}")
    
    # Group importance by category
    groups = {}
    for name, g in pairs:
        if name.startswith("aef_diff"): cat = "aef_diff"
        elif name.startswith("aef_base"): cat = "aef_base"
        elif name.startswith("aef_"): cat = "aef_scalar"
        elif name.startswith("s2_raw"): cat = "s2_raw_bands"
        elif name.startswith("s2_"): cat = "s2_indices"
        elif name.startswith("s1_"): cat = "s1_vv"
        else: cat = "other"
        groups.setdefault(cat, []).append(g)
    
    print("\nImportance by category (total gain):")
    for cat, vals in sorted(groups.items(), key=lambda x: -sum(x[1])):
        print(f"  {cat:>20s}: total={sum(vals):>12.1f}  mean={np.mean(vals):>10.1f}  n={len(vals)}")
    
    # Stat-type importance across all signals
    stat_groups = {}
    for name, g in pairs:
        for stat in _STAT_NAMES + ["trend"] + _CHG_NAMES:
            if name.endswith(f"_{stat}"):
                stat_groups.setdefault(stat, []).append(g)
                break
    
    print("\nImportance by stat type (mean gain across all signals):")
    for stat, vals in sorted(stat_groups.items(), key=lambda x: -np.mean(x[1])):
        print(f"  {stat:>15s}: mean={np.mean(vals):>10.1f}  total={sum(vals):>12.1f}  n={len(vals)}")
    
    # Zero-importance features  
    zero_feats = [n for n, g in pairs if g == 0]
    print(f"\nZero-importance features: {len(zero_feats)}")
    if zero_feats:
        for n in zero_feats[:30]:
            print(f"  {n}")

else:
    print("No model found, skipping feature importance")

# ═══════════════════════════════════════════════════════════════
# 2) Data availability per tile
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2) DATA AVAILABILITY PER TILE")
print("=" * 60)

for split in ["train", "test"]:
    tiles = list_tile_ids(split)
    print(f"\n{split.upper()} ({len(tiles)} tiles):")
    for tid in tiles:
        s2 = list_s2_timesteps(tid, split)
        s1 = list_s1_timesteps(tid, split)
        aef = list_aef_years(tid, split)
        s2_years = sorted(set(y for y, m in s2))
        s2_pre = len([1 for y, m in s2 if y <= DEFORESTATION_CUTOFF_YEAR])
        s2_post = len([1 for y, m in s2 if y > DEFORESTATION_CUTOFF_YEAR])
        print(f"  {tid}: S2={len(s2)} (pre={s2_pre}, post={s2_post}), S1={len(s1)}, AEF={aef}, S2_years={s2_years}")

# ═══════════════════════════════════════════════════════════════
# 3) Label statistics & source agreement
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3) LABEL STATISTICS & SOURCE AGREEMENT")
print("=" * 60)

train_tiles = list_tile_ids("train")
for tid in train_tiles:
    # Get reference grid from S2
    s2_steps = list_s2_timesteps(tid, "train")
    if not s2_steps:
        print(f"  {tid}: no S2 data")
        continue
    ref_data, ref_meta = load_s2(tid, s2_steps[0][0], s2_steps[0][1], "train")
    ref_transform, ref_crs, ref_shape = ref_meta["transform"], ref_meta["crs"], ref_data.shape[1:]
    
    sources = {}
    # RADD
    try:
        radd_raw, radd_meta = load_radd(tid)
        radd_binary, _ = decode_radd(radd_raw, min_confidence=RADD_CONFIDENCE_THRESHOLD, after_year=DEFORESTATION_CUTOFF_YEAR)
        if radd_meta["crs"] != ref_crs or radd_meta["shape"] != ref_shape:
            radd_binary = reproject_to_target(radd_binary, radd_meta["transform"], radd_meta["crs"],
                                              ref_transform, ref_crs, ref_shape, Resampling.nearest)
        sources["radd"] = radd_binary
    except: pass
    
    # GLAD-S2
    try:
        gs2_alert, gs2_date, gs2_meta = load_glads2(tid)
        gs2_binary, _ = decode_glads2(gs2_alert, gs2_date, min_confidence=GLADS2_MIN_CONFIDENCE, after_year=DEFORESTATION_CUTOFF_YEAR)
        if gs2_meta["crs"] != ref_crs or gs2_meta["shape"] != ref_shape:
            gs2_binary = reproject_to_target(gs2_binary, gs2_meta["transform"], gs2_meta["crs"],
                                             ref_transform, ref_crs, ref_shape, Resampling.nearest)
        sources["glads2"] = gs2_binary
    except: pass
    
    # GLAD-L
    gladl_combined = None
    for yy in range(21, 26):
        try:
            gl_alert, gl_date, gl_meta = load_gladl(tid, str(yy))
            gl_binary, _ = decode_gladl(gl_alert, gl_date, 2000 + yy, min_confidence=2, after_year=DEFORESTATION_CUTOFF_YEAR)
            if gl_meta["crs"] != ref_crs or gl_meta["shape"] != ref_shape:
                gl_binary = reproject_to_target(gl_binary, gl_meta["transform"], gl_meta["crs"],
                                                ref_transform, ref_crs, ref_shape, Resampling.nearest)
            gladl_combined = gl_binary if gladl_combined is None else np.maximum(gladl_combined, gl_binary)
        except: continue
    if gladl_combined is not None:
        sources["gladl"] = gladl_combined
    
    n_srcs = len(sources)
    total = ref_shape[0] * ref_shape[1]
    
    print(f"\n  {tid} ({ref_shape}, {n_srcs} label sources):")
    for sname, sbin in sources.items():
        pos = sbin.sum()
        print(f"    {sname:>8s}: {pos:>8d} positive ({100*pos/total:.3f}%)")
    
    if n_srcs >= 2:
        # Agreement analysis
        stack = np.stack(list(sources.values()), axis=0)
        vote_sum = stack.sum(axis=0)
        any_positive = (vote_sum > 0).sum()
        all_agree = (vote_sum == n_srcs).sum()
        maj = (vote_sum >= 2).sum()
        print(f"    ANY source positive: {any_positive} ({100*any_positive/total:.3f}%)")
        print(f"    Majority (>=2) agree: {maj} ({100*maj/total:.3f}%)")
        print(f"    ALL agree: {all_agree} ({100*all_agree/total:.3f}%)")

# ═══════════════════════════════════════════════════════════════
# 4) Class balance in current training data
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4) CLASS BALANCE ANALYSIS")
print("=" * 60)

# Re-run build to check class ratios
from solution.run_pipeline import build_features_for_tile, build_labels_for_tile
from solution.src.data.dataset import sample_pixels

total_pos = 0
total_neg = 0
for tid in train_tiles:
    result = build_features_for_tile(tid, "train")
    if result is None: continue
    features, ref_meta = result
    labels = build_labels_for_tile(tid, ref_meta)
    if labels is None: continue
    pos = labels.sum()
    neg = labels.size - pos
    total_pos += pos
    total_neg += neg
    pct = 100 * pos / labels.size
    print(f"  {tid}: pos={pos:>8d}, neg={neg:>8d}, ratio={pct:.2f}%")
    del features, labels, result
    gc.collect()

print(f"\n  TOTAL: pos={total_pos}, neg={total_neg}, ratio={100*total_pos/(total_pos+total_neg):.2f}%")
print(f"  Imbalance ratio: 1:{total_neg // max(total_pos,1)}")

# ═══════════════════════════════════════════════════════════════
# 5) Threshold optimization on val set
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5) THRESHOLD OPTIMIZATION")
print("=" * 60)

if model_path.exists():
    # Rebuild training data to do threshold sweep
    from solution.src.models.baseline_gbm import predict_lightgbm
    
    all_X, all_y = [], []
    for i, tid in enumerate(train_tiles):
        result = build_features_for_tile(tid, "train")
        if result is None: continue
        features, ref_meta = result
        labels = build_labels_for_tile(tid, ref_meta)
        if labels is None: continue
        X_tile, y_tile = sample_pixels(features, labels, sample_rate=PIXEL_SAMPLE_RATE, seed=RANDOM_SEED + i)
        all_X.append(X_tile)
        all_y.append(y_tile)
        del features, labels, result
        gc.collect()
    
    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)
    
    rng = np.random.default_rng(RANDOM_SEED)
    n = X_all.shape[0]
    idx = rng.permutation(n)
    split_at = int(0.8 * n)
    X_val = X_all[idx[split_at:]]
    y_val = y_all[idx[split_at:]]
    
    y_prob = predict_lightgbm(model, X_val)
    
    print("\nThreshold sweep on validation set:")
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.20, 0.70, 0.02):
        y_pred = (y_prob >= t).astype(np.uint8)
        tp = ((y_pred == 1) & (y_val == 1)).sum()
        fp = ((y_pred == 1) & (y_val == 0)).sum()
        fn = ((y_pred == 0) & (y_val == 1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        marker = " <-- BEST" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1, best_t = f1, t
        print(f"  t={t:.2f}: F1={f1:.4f} P={prec:.4f} R={rec:.4f}{marker}")
    
    print(f"\n  Best threshold: {best_t:.2f} → F1={best_f1:.4f}")
    
    # Also check scale_pos_weight impact estimate
    neg_count = (y_all == 0).sum()
    pos_count = (y_all == 1).sum()
    print(f"\n  Suggested scale_pos_weight: {neg_count / max(pos_count, 1):.1f}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
