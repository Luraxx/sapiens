# Solution — Deforestation Detection

This is our solution for the osapiens Makeathon 2026 challenge.

See [PLAN.md](PLAN.md) for the full strategy and execution plan.

## Project Structure

```
solution/
├── PLAN.md                  # Detailed solution plan and strategy
├── README.md                # This file
├── config.py                # Global paths, constants, hyperparameters
├── run_pipeline.py          # End-to-end: features → train → predict → submit
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py        # Load S1, S2, AEF, labels by tile ID
│   │   ├── reproject.py     # CRS alignment utilities
│   │   ├── inventory.py     # Scan data dir, list tiles & available time steps
│   │   └── dataset.py       # PyTorch Dataset / numpy array builder
│   │
│   ├── labels/
│   │   ├── __init__.py
│   │   ├── decode.py        # Decode RADD, GLAD-L, GLAD-S2 raw encodings
│   │   └── fuse.py          # Combine weak labels into pseudo ground truth
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── indices.py       # NDVI, EVI, NBR, NDMI from S2 bands
│   │   ├── temporal.py      # Temporal stats, trend, change detection
│   │   └── embeddings.py    # AEF embedding processing
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_gbm.py  # LightGBM / XGBoost pixel classifier
│   │   └── unet.py          # U-Net segmentation model
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_gbm.py     # Train tabular model
│   │   ├── train_unet.py    # Train U-Net
│   │   └── cross_val.py     # Spatial cross-validation splits
│   │
│   └── inference/
│       ├── __init__.py
│       ├── predict.py       # Run model on test tiles
│       └── submit.py        # Convert predictions → GeoJSON submission
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_label_fusion.ipynb
│   ├── 03_baseline.ipynb
│   └── 04_unet.ipynb
│
└── outputs/
    ├── predictions/          # Binary GeoTIFFs per test tile
    └── submissions/          # Final GeoJSON files for leaderboard
```

## Quick Start

```bash
# From the repo root
make install
make download_data_from_s3

# Run the baseline pipeline
cd solution
python run_pipeline.py
```
