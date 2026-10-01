"""Run a small synthetic multimodal robustness experiment.

This experiment is a feature-level software prototype. It does not use real
camera/LiDAR/RADAR data and is not a 3D-object-detection benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from mfgds.devices import resolve_device

from mfgds.sensor_fusion import (
    MODALITIES,
    FusionClassifier,
    MissingModalityCompleter,
    SyntheticSensorConfig,
    accuracy_from_logits,
    apply_modality_dropout,
    macro_f1_from_logits,
    make_synthetic_sensor_dataset,
    restore_missing_modality,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_modalities(modalities, split_index):
    train = {k: v[:split_index] for k, v in modalities.items()}
    test = {k: v[split_index:] for k, v in modalities.items()}
    return train, test


def train_classifier(model, x, y, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()


def train_completer(model, x, target, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), target)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(classifier, x, y):
    classifier.eval()
    logits = classifier(x)
    return {
        "accuracy": accuracy_from_logits(logits, y),
        "macro_f1": macro_f1_from_logits(logits, y),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--completion-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", help="cuda (default), cuda:N, cpu or auto")
    parser.add_argument("--output", type=Path, default=Path("results/sensor_fusion_robustness.json"))
    args = parser.parse_args()

    if args.samples < 20:
        raise ValueError("--samples must be at least 20")
    device = resolve_device(args.device)
    print(f"Compute device: {device}", flush=True)

    set_seed(args.seed)
    cfg = SyntheticSensorConfig()
    modalities, labels = make_synthetic_sensor_dataset(args.samples, cfg, args.seed)
    modalities = {k: v.to(device) for k, v in modalities.items()}
    labels = labels.to(device)

    split = int(args.samples * 0.75)
    train_x, test_x = split_modalities(modalities, split)
    train_y, test_y = labels[:split], labels[split:]

    classifier = FusionClassifier(cfg.dims, cfg.num_classes).to(device)
    train_classifier(classifier, train_x, train_y, args.epochs, lr=1e-2)

    report = {
        "note": (
            "Synthetic feature-level proxy only; not real sensor data, BEV geometry, "
            "3D detection, or adverse-weather validation."
        ),
        "seed": args.seed,
        "device": str(device),
        "samples": args.samples,
        "all_modalities": evaluate(classifier, test_x, test_y),
        "dropout": {},
        "completion_assisted": {},
    }

    for modality in MODALITIES:
        degraded_test = apply_modality_dropout(test_x, modality)
        report["dropout"][modality] = evaluate(classifier, degraded_test, test_y)

        completer = MissingModalityCompleter(cfg.dims, modality).to(device)
        train_completer(
            completer,
            train_x,
            train_x[modality],
            args.completion_epochs,
            lr=1e-2,
        )
        completer.eval()
        restored = restore_missing_modality(degraded_test, completer)
        report["completion_assisted"][modality] = evaluate(
            classifier,
            restored,
            test_y,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
