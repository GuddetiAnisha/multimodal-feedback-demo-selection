"""Synthetic multimodal sensor-fusion robustness research utilities.

This module is intentionally a software-only proxy for autonomous-driving
sensor fusion. It operates on synthetic feature vectors that are labelled as
camera-, LiDAR-, and RADAR-like modalities. It does not claim real sensor
physics, bird's-eye-view geometry, 3D detection, or adverse-weather realism.

The goal is to provide a reproducible testbed for:
- multimodal feature fusion,
- modality dropout,
- learned missing-modality completion,
- graceful-degradation evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
from torch import nn

MODALITIES: Tuple[str, ...] = ("camera", "lidar", "radar")


@dataclass(frozen=True)
class SyntheticSensorConfig:
    """Configuration for the synthetic multimodal feature generator."""

    camera_dim: int = 16
    lidar_dim: int = 12
    radar_dim: int = 8
    latent_dim: int = 6
    num_classes: int = 3
    noise_std: float = 0.15

    @property
    def dims(self) -> Dict[str, int]:
        return {
            "camera": self.camera_dim,
            "lidar": self.lidar_dim,
            "radar": self.radar_dim,
        }


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_synthetic_sensor_dataset(
    n_samples: int = 512,
    config: SyntheticSensorConfig | None = None,
    seed: int = 0,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Create correlated synthetic modality features and class labels.

    A shared latent scene vector is projected into three independent feature
    spaces. Labels are derived from another fixed projection of the same
    latent state. This produces a controlled benchmark where modalities are
    complementary and correlated without pretending to simulate real sensors.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    cfg = config or SyntheticSensorConfig()
    if cfg.num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    rng = _rng(seed)
    latent = rng.normal(size=(n_samples, cfg.latent_dim)).astype(np.float32)

    modalities: Dict[str, torch.Tensor] = {}
    for index, (name, dim) in enumerate(cfg.dims.items()):
        projection_rng = _rng(seed + 101 + index)
        projection = projection_rng.normal(
            scale=1.0 / np.sqrt(cfg.latent_dim),
            size=(cfg.latent_dim, dim),
        ).astype(np.float32)
        noise = rng.normal(
            scale=cfg.noise_std,
            size=(n_samples, dim),
        ).astype(np.float32)
        features = latent @ projection + noise
        modalities[name] = torch.from_numpy(features)

    label_rng = _rng(seed + 999)
    label_projection = label_rng.normal(
        scale=1.0 / np.sqrt(cfg.latent_dim),
        size=(cfg.latent_dim, cfg.num_classes),
    ).astype(np.float32)
    logits = latent @ label_projection
    labels = torch.from_numpy(logits.argmax(axis=1).astype(np.int64))
    return modalities, labels


def apply_modality_dropout(
    modalities: Mapping[str, torch.Tensor],
    missing: str | Iterable[str],
) -> Dict[str, torch.Tensor]:
    """Return a copy with one or more modalities replaced by zeros."""

    missing_set = {missing} if isinstance(missing, str) else set(missing)
    unknown = missing_set.difference(modalities)
    if unknown:
        raise KeyError(f"unknown modalities: {sorted(unknown)}")

    return {
        name: torch.zeros_like(value) if name in missing_set else value.clone()
        for name, value in modalities.items()
    }


def concatenate_modalities(
    modalities: Mapping[str, torch.Tensor],
    order: Iterable[str] = MODALITIES,
) -> torch.Tensor:
    """Concatenate modality features in a stable order."""

    tensors = []
    batch_size = None
    for name in order:
        if name not in modalities:
            raise KeyError(f"missing modality: {name}")
        tensor = modalities[name]
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, features]")
        if batch_size is None:
            batch_size = tensor.shape[0]
        elif tensor.shape[0] != batch_size:
            raise ValueError("all modalities must share the same batch size")
        tensors.append(tensor)
    return torch.cat(tensors, dim=1)


class FusionClassifier(nn.Module):
    """Small feature-level fusion classifier used for robustness experiments."""

    def __init__(
        self,
        dims: Mapping[str, int],
        num_classes: int,
        hidden_dim: int = 48,
    ) -> None:
        super().__init__()
        input_dim = sum(int(dims[name]) for name in MODALITIES)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, modalities: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.network(concatenate_modalities(modalities))


class MissingModalityCompleter(nn.Module):
    """Predict one missing modality from the remaining two modalities."""

    def __init__(
        self,
        dims: Mapping[str, int],
        target_modality: str,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if target_modality not in MODALITIES:
            raise ValueError(f"target_modality must be one of {MODALITIES}")

        self.target_modality = target_modality
        self.source_modalities = tuple(
            name for name in MODALITIES if name != target_modality
        )
        source_dim = sum(int(dims[name]) for name in self.source_modalities)
        target_dim = int(dims[target_modality])
        self.network = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, modalities: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.network(concatenate_modalities(modalities, self.source_modalities))


def restore_missing_modality(
    degraded: Mapping[str, torch.Tensor],
    completer: MissingModalityCompleter,
) -> Dict[str, torch.Tensor]:
    """Insert a learned reconstruction into a degraded modality dictionary."""

    restored = {name: value.clone() for name, value in degraded.items()}
    restored[completer.target_modality] = completer(degraded)
    return restored


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return classification accuracy for logits and integer labels."""

    if logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("expected logits [batch, classes] and labels [batch]")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must share the same batch size")
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def macro_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute macro F1 without an external metrics dependency."""

    predictions = logits.argmax(dim=1)
    num_classes = logits.shape[1]
    scores = []
    for cls in range(num_classes):
        tp = ((predictions == cls) & (labels == cls)).sum().item()
        fp = ((predictions == cls) & (labels != cls)).sum().item()
        fn = ((predictions != cls) & (labels == cls)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        scores.append(f1)
    return float(np.mean(scores))
