# Solution Plan — Detecting Deforestation from Space

## 1. Challenge Summary

| Aspect | Detail |
|--------|--------|
| **Goal** | Binary pixel-level deforestation detection (post-2020) on unseen test tiles |
| **Input** | Sentinel-2 (12 bands, monthly), Sentinel-1 (1 band VV, monthly, ascending+descending), AlphaEarth Foundation embeddings (64 dims, annual) |
| **Labels** | 3 weak/noisy sources: RADD (radar), GLAD-L (Landsat), GLAD-S2 (Sentinel-2) |
| **Output** | Binary GeoTIFF per test tile → converted to GeoJSON via `submission_utils.py` |
| **Evaluation** | Quantitative (hidden test set) + qualitative (jury: design, generalization, presentation) |
| **CRS caveat** | S1 & S2 are in UTM; AEF & labels are in EPSG:4326 — **must reproject before combining** |

---

## 2. Key Technical Challenges

1. **Noisy / conflicting labels** — All 3 label sources are weak predictions, not ground truth. They disagree.
2. **Multimodal fusion** — Radar (all-weather) vs. optical (cloud-affected) vs. foundation embeddings (high-level features).
3. **Temporal reasoning** — Deforestation is a *change*: forest in 2020 → non-forest later. Need before/after comparison.
4. **CRS alignment** — Labels & AEF in EPSG:4326; S1/S2 in UTM. Must reproject to common grid.
5. **Generalization** — Test tiles may be in different biomes/countries than training tiles.
6. **Scale** — Large rasters, many time steps. Memory-efficient pipelines needed.

---

## 3. Solution Architecture (5 Phases)

### Phase 1: Data Engineering & EDA
> Understand the data before modeling.

- [ ] Download data (`make download_data_from_s3`)
- [ ] Inventory: list all tiles, count time steps per tile, check for missing months
- [ ] CRS alignment: build reproject utility (all to UTM per tile)
- [ ] EDA notebook: visualize each modality, overlay labels, check label agreement/disagreement
- [ ] Compute basic stats: band distributions, no-data percentages, cloud coverage proxies (S2 B10 cirrus)

**Deliverable**: `notebooks/01_eda.ipynb`, `src/data/reproject.py`

### Phase 2: Label Fusion (Pseudo Ground Truth)
> Combine 3 weak label sources into a single, higher-quality supervision signal.

**Strategy options (try in order of complexity)**:
1. **Majority vote** — pixel is deforested if ≥2 of 3 sources agree (after thresholding confidence)
2. **Confidence-weighted union** — weight each source by its confidence level, threshold the weighted sum
3. **Learned label fusion** (Snorkel-style) — train a small model to learn label accuracies per source and fuse probabilistically

**Label preprocessing**:
- RADD: decode `leading digit` for confidence (2=low, 3=high), extract date from remaining digits, filter post-2020
- GLAD-L: for each year YY (21–25), take `alert ∈ {2,3}` as positive, use alertDate for temporal info
- GLAD-S2: take `alert ∈ {2,3,4}` as positive (skip 1=recent-only as too noisy), use alertDate for temporal info
- Reproject all labels to UTM grid of each tile

**Deliverable**: `src/labels/decode.py`, `src/labels/fuse.py`, `notebooks/02_label_fusion.ipynb`

### Phase 3: Feature Engineering & Dataset
> Create ML-ready features from raw satellite data.

**Sentinel-2 features** (per pixel, temporal):
- Vegetation indices: NDVI, EVI, NBR, NDMI from band math
- Temporal statistics: mean, std, min, max, slope (linear trend) per index across months
- Change features: Δ(post-2020 mean − 2020 baseline) for each index
- Cloud masking: use B10 (cirrus) + SCL if available to mask bad observations

**Sentinel-1 features** (per pixel, temporal):
- Convert linear to dB
- Temporal statistics: mean, std, min, max, percentiles
- Change features: Δ(post-2020 − 2020 baseline)
- Ascending vs. descending: treat separately or average

**AlphaEarth Foundation embeddings**:
- Reproject to UTM
- Use raw 64-dim vectors as features
- Temporal change: diff(2021+ embedding − 2020 embedding) for each dimension

**Dataset construction**:
- Option A: **Pixel-level flat features** → tabular model (XGBoost/LightGBM) — fast baseline
- Option B: **Patch-based** (e.g. 64×64 patches) → CNN/U-Net — spatial context
- Option C: **Pixel time series** → temporal model (1D-CNN, LSTM, Transformer)

**Deliverable**: `src/features/indices.py`, `src/features/temporal.py`, `src/features/embeddings.py`, `src/data/dataset.py`

### Phase 4: Modeling
> Train, validate, evaluate.

**Approach 1: Strong Baseline — Pixel-Level Gradient Boosting**
- Flatten all temporal features + embeddings into a single feature vector per pixel
- Train LightGBM / XGBoost on fused labels
- Fast iteration, strong baseline, interpretable feature importances
- Spatial cross-validation: hold out entire tiles (not random pixels) to simulate test conditions

**Approach 2: Spatial Model — U-Net on Patches**
- Input: multi-channel patch (stacked temporal features + embeddings)
- Output: binary segmentation mask
- Architecture: U-Net with ResNet encoder (pretrained or from scratch)
- Data augmentation: flips, rotations, random crops

**Approach 3: Temporal Model (if time permits)**
- Per-pixel time series classification (LSTM / Temporal CNN / Transformer)
- Input: monthly S1+S2 bands + annual embeddings
- Can capture the temporal signature of deforestation

**Validation strategy**:
- **Spatial k-fold**: split by tile ID, ensure geographic diversity in each fold
- **Metric**: F1-score (binary), IoU, and precision/recall (for understanding false positives vs. missed events)

**Deliverable**: `src/models/`, `src/training/`, `notebooks/03_baseline.ipynb`, `notebooks/04_unet.ipynb`

### Phase 5: Inference & Submission
> Generate predictions for test tiles, convert to GeoJSON.

- Run trained model on all test tiles
- Post-processing: morphological operations (remove tiny isolated pixels), minimum area filter
- Convert binary rasters → GeoJSON using `submission_utils.py`
- Concatenate all tile GeoJSONs into single submission file

**Deliverable**: `src/inference/predict.py`, `src/inference/submit.py`

---

## 4. Bonus Tasks

| Bonus | Approach |
|-------|----------|
| **When** deforestation occurred | Use alertDate from labels + temporal feature analysis to predict month/year |
| **Confidence estimation** | Output probability instead of binary; calibrate with Platt scaling |
| **Visualization tool** | Streamlit/Gradio app: select tile → show S2 RGB + prediction overlay + confidence heatmap |

---

## 5. Execution Priority (Hackathon Time Management)

| Priority | Task | Est. Impact |
|----------|------|-------------|
| **P0** | Data download + EDA + label understanding | Foundation |
| **P0** | Label fusion (majority vote) | Supervision quality |
| **P0** | Pixel-level feature engineering + LightGBM baseline | First submission |
| **P1** | Improved label fusion (confidence-weighted) | Better labels → better model |
| **P1** | U-Net spatial model | Likely best performance |
| **P2** | Temporal model | Harder but captures change dynamics |
| **P2** | Ensemble of approaches | Combine strengths |
| **P3** | Bonus tasks (when, confidence, viz) | Jury points |

---

## 6. Tech Stack

| Component | Tool |
|-----------|------|
| Data loading | `rasterio`, `geopandas`, `numpy` |
| Reprojection | `rasterio.warp.reproject` |
| Feature engineering | `numpy`, `scipy` |
| Tabular ML | `lightgbm` or `xgboost` |
| Deep learning | `pytorch`, `segmentation-models-pytorch` |
| Experiment tracking | `wandb` or simple CSV logs |
| Visualization | `matplotlib`, `folium` |
| Submission | `submission_utils.py` (provided) |

---

## 7. Execution Results

### Model Iterations

| Version | Features | Sample | Rounds | Best Iter | F1 | Precision | Recall |
|---------|----------|--------|--------|-----------|-----|-----------|--------|
| v1 baseline | 54 (NDVI/NBR/NDMI + S1 + AEF) | 5% | 1000 | 701 | 0.810 | 0.820 | 0.800 |
| v4 +EVI +raw bands | 77 | 5% | 1000 | 872 | 0.815 | 0.826 | 0.803 |
| v5 +10% sampling | 77 | 10% | 1000 | 997 | 0.828 | 0.840 | 0.816 |
| v6 +AEF yoy change | 79 | 10% | 2000 | 1954 | 0.838 | 0.854 | 0.823 |
| **v7 +255 leaves** | **79** | **10%** | **3000** | **1078** | **0.839** | **0.856** | **0.822** |

### Top Feature Importances (gain)
1. `aef_l2_dist` (AEF embedding L2 distance baseline→latest) — **by far #1**
2. `aef_cosine_dist` (AEF embedding cosine distance)
3. `aef_base_g4` (AEF pooled group 4)
4. `s2_ndmi_delta_min` (NDMI min change post vs baseline)
5. `s2_raw_swir1_mean` (raw SWIR1 band mean)

### Test Predictions
| Tile | Region | Defor Pixels | % | Polygons |
|------|--------|-------------|---|----------|
| 18NVJ_1_6 | S. America | 0 | 0.00% | 0 |
| 18NYH_2_1 | S. America | 53,155 | 5.27% | 122 |
| 33NTE_5_1 | Africa | 8,322 | 0.82% | 35 |
| 47QMA_6_2 | SE Asia | 0 | 0.00% | 0 |
| 48PWA_0_6 | SE Asia | 12,066 | 1.23% | 66 |

**Submission**: 223 polygons in GeoJSON (EPSG:4326)

### Key Insights
- Foundation model embeddings (AEF) are the most discriminative features — they capture land cover change extremely well
- Morphological post-processing (close→open→remove small components) improves polygon quality
- Optimal threshold ≈ 0.45 (slight improvement over 0.5)
- Two tiles (18NVJ, 47QMA) show 0 deforestation — model is confident, may be genuinely forest-stable areas
