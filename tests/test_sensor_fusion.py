import torch

from mfgds.sensor_fusion import (
    FusionClassifier,
    MissingModalityCompleter,
    SyntheticSensorConfig,
    accuracy_from_logits,
    apply_modality_dropout,
    concatenate_modalities,
    macro_f1_from_logits,
    make_synthetic_sensor_dataset,
    restore_missing_modality,
)


def test_synthetic_dataset_is_deterministic():
    cfg = SyntheticSensorConfig()
    first_x, first_y = make_synthetic_sensor_dataset(32, cfg, seed=7)
    second_x, second_y = make_synthetic_sensor_dataset(32, cfg, seed=7)

    assert torch.equal(first_y, second_y)
    for name in cfg.dims:
        assert torch.allclose(first_x[name], second_x[name])


def test_modality_dropout_preserves_other_modalities():
    data, _ = make_synthetic_sensor_dataset(8, seed=1)
    degraded = apply_modality_dropout(data, "lidar")

    assert torch.count_nonzero(degraded["lidar"]) == 0
    assert torch.allclose(degraded["camera"], data["camera"])
    assert torch.allclose(degraded["radar"], data["radar"])


def test_fusion_and_completion_shapes():
    cfg = SyntheticSensorConfig(camera_dim=10, lidar_dim=7, radar_dim=5)
    data, _ = make_synthetic_sensor_dataset(16, cfg, seed=3)

    classifier = FusionClassifier(cfg.dims, cfg.num_classes)
    logits = classifier(data)
    assert logits.shape == (16, cfg.num_classes)

    completer = MissingModalityCompleter(cfg.dims, "radar")
    degraded = apply_modality_dropout(data, "radar")
    reconstruction = completer(degraded)
    assert reconstruction.shape == data["radar"].shape

    restored = restore_missing_modality(degraded, completer)
    assert restored["radar"].shape == data["radar"].shape


def test_concatenation_width_matches_configuration():
    cfg = SyntheticSensorConfig(camera_dim=4, lidar_dim=3, radar_dim=2)
    data, _ = make_synthetic_sensor_dataset(5, cfg, seed=4)
    combined = concatenate_modalities(data)
    assert combined.shape == (5, 9)


def test_metrics_are_bounded():
    logits = torch.tensor([[3.0, 0.1], [0.2, 2.0], [2.0, 0.5]])
    labels = torch.tensor([0, 1, 1])

    accuracy = accuracy_from_logits(logits, labels)
    f1 = macro_f1_from_logits(logits, labels)

    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= f1 <= 1.0
