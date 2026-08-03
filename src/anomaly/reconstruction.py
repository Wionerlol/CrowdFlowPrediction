"""Reconstruction-error anomaly detectors for Week 5.

This module provides two unsupervised reconstruction baselines:

- a vector VAE over one flattened flow grid;
- a lightweight Transformer autoencoder over short flow sequences.

Both models operate on per-node standardized inflow values and return
node-level absolute reconstruction errors in standardized units.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


Array = np.ndarray


@dataclass(frozen=True)
class NodeStandardScaler:
    """Per-node standardization fitted on training inflow."""

    mean: Array
    std: Array

    @classmethod
    def fit(cls, x: Array, *, min_std: float = 1.0) -> "NodeStandardScaler":
        arr = np.asarray(x, dtype=np.float32)
        mean = arr.mean(axis=0)
        std = np.maximum(arr.std(axis=0), min_std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: Array) -> Array:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x: Array) -> Array:
        return (np.asarray(x, dtype=np.float32) * self.std + self.mean).astype(np.float32)


class VectorDataset(Dataset):
    """Simple vector dataset for VAE training."""

    def __init__(self, x: Array):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


class SequenceDataset(Dataset):
    """Sliding windows ending at each available timestamp."""

    def __init__(self, x: Array, seq_len: int, valid_starts: Array | None = None):
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        arr = np.asarray(x, dtype=np.float32)
        if len(arr) < seq_len:
            raise ValueError(f"Need at least seq_len={seq_len} rows, got {len(arr)}")
        self.x = torch.from_numpy(arr)
        self.seq_len = int(seq_len)
        max_start = len(arr) - self.seq_len
        if valid_starts is None:
            self.valid_starts = np.arange(max_start + 1, dtype=np.int64)
        else:
            starts = np.asarray(valid_starts, dtype=np.int64)
            if starts.ndim != 1:
                raise ValueError("valid_starts must be a 1D array")
            if len(starts) == 0:
                raise ValueError("valid_starts is empty")
            if starts.min() < 0 or starts.max() > max_start:
                raise ValueError(
                    f"valid_starts must be within [0, {max_start}], got "
                    f"[{starts.min()}, {starts.max()}]"
                )
            self.valid_starts = starts

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = int(self.valid_starts[idx])
        return self.x[start:start + self.seq_len]


def normal_sequence_starts(labels: Array, seq_len: int) -> Array:
    """Return sequence starts whose full window contains no weak anomaly."""

    labels = np.asarray(labels, dtype=bool)
    if len(labels) < seq_len:
        raise ValueError(f"Need at least seq_len={seq_len} labels, got {len(labels)}")
    bad = labels.astype(np.int32)
    prefix = np.concatenate([[0], np.cumsum(bad)])
    counts = prefix[seq_len:] - prefix[:-seq_len]
    return np.flatnonzero(counts == 0).astype(np.int64)


class VAE(nn.Module):
    """Small fully-connected VAE for flattened flow grids."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class TransformerAutoEncoder(nn.Module):
    """Lightweight Transformer sequence autoencoder.

    It compresses each flattened grid through a linear projection, applies a
    standard Transformer encoder over time, and reconstructs the full grid at
    each timestep.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 12,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos[:, :x.shape[1]]
        h = self.encoder(h)
        return self.output(h)


def _limit_batches(loader: DataLoader, max_batches: int | None) -> Iterable:
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        yield batch


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1e-4,
) -> torch.Tensor:
    mse = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kl


def train_vae(
    train_x: Array,
    val_x: Array,
    *,
    hidden_dim: int = 256,
    latent_dim: int = 32,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    epochs: int = 20,
    patience: int = 5,
    beta: float = 1e-4,
    device: torch.device | str = "cpu",
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> tuple[VAE, dict[str, float | int]]:
    """Train a VAE and return the best validation checkpoint in memory."""

    device = torch.device(device)
    input_dim = int(train_x.shape[1])
    model = VAE(input_dim, hidden_dim, latent_dim, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(VectorDataset(train_x), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(VectorDataset(val_x), batch_size=batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    bad = 0
    last_epoch = 0
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()
        train_losses = []
        for batch in _limit_batches(train_loader, max_train_batches):
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in _limit_batches(val_loader, max_val_batches):
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                val_losses.append(float(vae_loss(recon, batch, mu, logvar, beta=beta).cpu()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best_val, "epochs_ran": last_epoch}


def train_transformer_autoencoder(
    train_x: Array,
    val_x: Array,
    *,
    train_valid_starts: Array | None = None,
    seq_len: int = 12,
    d_model: int = 64,
    n_heads: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 15,
    patience: int = 4,
    device: torch.device | str = "cpu",
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> tuple[TransformerAutoEncoder, dict[str, float | int]]:
    """Train a Transformer autoencoder on normalized flow sequences."""

    device = torch.device(device)
    input_dim = int(train_x.shape[1])
    model = TransformerAutoEncoder(
        input_dim=input_dim,
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_ds = SequenceDataset(train_x, seq_len, valid_starts=train_valid_starts)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(val_x, seq_len), batch_size=batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    bad = 0
    last_epoch = 0
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()
        for batch in _limit_batches(train_loader, max_train_batches):
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in _limit_batches(val_loader, max_val_batches):
                batch = batch.to(device)
                val_losses.append(float(F.mse_loss(model(batch), batch).cpu()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_val_loss": best_val,
        "epochs_ran": last_epoch,
        "train_sequence_count": len(train_ds),
    }


@torch.no_grad()
def score_vae(
    model: VAE,
    x: Array,
    *,
    batch_size: int = 512,
    device: torch.device | str = "cpu",
) -> Array:
    """Return standardized absolute reconstruction error, shape ``(T,N)``."""

    device = torch.device(device)
    model = model.to(device)
    model.eval()
    loader = DataLoader(VectorDataset(x), batch_size=batch_size, shuffle=False)
    errors = []
    for batch in loader:
        batch = batch.to(device)
        recon, _, _ = model(batch)
        errors.append((recon - batch).abs().cpu().numpy())
    return np.concatenate(errors, axis=0).astype(np.float32)


@torch.no_grad()
def score_transformer_targets(
    model: TransformerAutoEncoder,
    series_x: Array,
    target_indices: Array,
    *,
    seq_len: int,
    batch_size: int = 128,
    device: torch.device | str = "cpu",
) -> Array:
    """Score selected target timestamps using windows ending at each target.

    ``series_x`` must be the full normalized time series. ``target_indices`` are
    absolute row indices into that series.
    """

    device = torch.device(device)
    model = model.to(device)
    model.eval()
    arr = np.asarray(series_x, dtype=np.float32)
    targets = np.asarray(target_indices, dtype=np.int64)
    if np.any(targets < seq_len - 1):
        raise ValueError("All target_indices must be >= seq_len - 1")

    errors = []
    for start in range(0, len(targets), batch_size):
        chunk = targets[start:start + batch_size]
        windows = np.stack([arr[i - seq_len + 1:i + 1] for i in chunk], axis=0)
        batch = torch.from_numpy(windows).to(device)
        recon = model(batch)
        err = (recon[:, -1] - batch[:, -1]).abs().cpu().numpy()
        errors.append(err)
    return np.concatenate(errors, axis=0).astype(np.float32)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    metadata: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metadata": metadata}, path)


def read_checkpoint_metadata(path: str | Path) -> dict:
    """Read checkpoint metadata without constructing a model."""

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return dict(ckpt.get("metadata", {}))


def load_vae_checkpoint(path: str | Path, *, input_dim: int, device: torch.device | str = "cpu") -> VAE:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt.get("metadata", {})
    model = VAE(
        input_dim=input_dim,
        hidden_dim=int(meta.get("hidden_dim", 256)),
        latent_dim=int(meta.get("latent_dim", 32)),
        dropout=float(meta.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device)


def load_transformer_checkpoint(
    path: str | Path,
    *,
    input_dim: int,
    device: torch.device | str = "cpu",
) -> TransformerAutoEncoder:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt.get("metadata", {})
    model = TransformerAutoEncoder(
        input_dim=input_dim,
        seq_len=int(meta.get("seq_len", 12)),
        d_model=int(meta.get("d_model", 64)),
        n_heads=int(meta.get("n_heads", 4)),
        num_layers=int(meta.get("num_layers", 2)),
        dim_feedforward=int(meta.get("dim_feedforward", 128)),
        dropout=float(meta.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device)
