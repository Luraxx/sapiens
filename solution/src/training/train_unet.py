"""Training loop for the U-Net segmentation model."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from solution.src.models.unet import UNet

logger = logging.getLogger(__name__)


def train_unet(
    X_patches: np.ndarray,
    y_patches: np.ndarray,
    in_channels: int,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "auto",
) -> UNet:
    """Train U-Net on patch data.

    Args:
        X_patches: (N, C, H, W) input patches.
        y_patches: (N, H, W) binary label patches.
        in_channels: Number of input channels.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        device: 'cpu', 'cuda', 'mps', or 'auto'.

    Returns:
        Trained UNet model.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = UNet(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.from_numpy(X_patches).float()
    y_t = torch.from_numpy(y_patches).float().unsqueeze(1)  # (N, 1, H, W)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")

    return model
