import copy
import numpy as np
import pytest
import torch
import yaml
from pathlib import Path
from mfgds.devices import cpu_snapshot, resolve_device
from mfgds.retrievers import UtilityRetriever, VectorIndex, pair_features
from mfgds.selectors import select, diversity


def test_cuda_missing_fails_clearly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="cpu, auto"):
        resolve_device("meta")


def test_snapshot_is_detached():
    tensor = torch.tensor([1.], requires_grad=True)
    saved = cpu_snapshot({"weights": tensor, "nested": [tensor]})
    with torch.no_grad():
        tensor.add_(10)
    assert saved["weights"].item() == 1
    assert not saved["weights"].requires_grad
    assert saved["nested"][0].device.type == "cpu"


def test_gpu_config_preserves_experiment_settings():
    base = yaml.safe_load(Path("configs/scienceqa.yaml").read_text())
    gpu = yaml.safe_load(Path("configs/scienceqa_gpu.yaml").read_text())
    assert gpu == base


def test_all_shipped_configs_require_cuda():
    for path in Path("configs").glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        assert config["device"] == "cuda"
        assert config["index_backend"] == "torch"
        if config["encoder"]["kind"] == "clip":
            assert config["encoder"]["device"] == "cuda"
        if config["model"]["kind"] == "hf":
            assert config["model"]["device"] == "cuda"


def test_default_run_never_falls_back_to_cpu(tmp_path, monkeypatch):
    from mfgds.experiment import run
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        run({}, tmp_path)


def test_global_device_applies_to_encoder_and_hf_model(tmp_path, monkeypatch):
    from mfgds import experiment
    from mfgds.embeddings import HashEncoder
    from mfgds.models import MockLMM
    seen = {}
    class Encoder(HashEncoder):
        def __init__(self, model, device):
            super().__init__()
            seen["encoder"] = device
    def model(**kwargs):
        seen["model"] = kwargs["device"]
        return MockLMM()
    monkeypatch.setattr(experiment, "ClipEncoder", Encoder)
    monkeypatch.setattr(experiment, "HuggingFaceLMM", model)
    config = {"device": "cpu", "index_backend": "torch",
              "encoder": {"kind": "clip", "device": "cuda"},
              "model": {"kind": "hf", "device": "cuda"},
              "dataset": {"kind": "synthetic", "sizes": [6, 3, 2]},
              "methods": ["feedback"], "seeds": [7], "ks": [0, 2],
              "orderings": ["best_first"], "context_budgets": [128],
              "feedback_candidates": 2, "epochs": 2}
    experiment.run(config, tmp_path)
    assert seen == {"encoder": "cpu", "model": "cpu"}


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware unavailable"))])
def test_torch_index_parity_and_ties(device):
    vectors = np.array([[1, 0], [.7, .7], [0, 1], [0, 0]], np.float32)
    q = np.array([1, .2], np.float32)
    expected = VectorIndex(vectors, "sklearn").search(q, 4)
    index = VectorIndex(vectors, "torch", device=device)
    actual = index.search(q, 4)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-6)
    assert index.tensor.device.type == device
    assert index.search(np.zeros(2), 4)[0].tolist() == [0, 1, 2, 3]
    assert len(index.search(q, 0)[0]) == 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware unavailable"))])
def test_training_device_and_resume(tmp_path, device):
    q = np.array([1, 0], np.float32)
    demos = np.array([[1, 0], [0, 1]], np.float32)
    features = pair_features(q, demos)
    expected = UtilityRetriever(2, seed=4, device=device)
    history = expected.fit(features, [1, -1], epochs=5)
    assert next(expected.net.parameters()).device.type == device
    partial = UtilityRetriever(2, seed=4, device=device)
    saved = []
    def interrupt(state):
        saved.append(copy.deepcopy(state))
        if state["epoch"] == 2:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        partial.fit(features, [1, -1], epochs=5, checkpoint=interrupt)
    resumed = UtilityRetriever(2, seed=4, device=device)
    actual = resumed.fit(features, [1, -1], epochs=5, resume_state=saved[-1])
    assert actual == history
    for key, value in expected.net.state_dict().items():
        assert torch.equal(value, resumed.net.state_dict()[key])
    assert saved[-1]["model"]["0.weight"].device.type == "cpu"
    resumed.save(tmp_path / "model.pt")
    loaded = UtilityRetriever.load(tmp_path / "model.pt", device=device)
    np.testing.assert_array_equal(loaded.score(q, demos), resumed.score(q, demos))
    assert next(UtilityRetriever.load(tmp_path / "model.pt", device="cpu").net.parameters()).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_gpu_selection_and_diversity():
    vectors = np.array([[1, 0], [1, 0], [0, 1]], np.float32)
    assert select([1, .95, .9], vectors, 2, .5, device="cuda") == [0, 2]
    assert select([1, 1, 1], vectors, 3, 0, device="cuda") == [0, 1, 2]
    assert select([1, 2, 3], vectors, 0, device="cuda") == []
    np.testing.assert_allclose(diversity(vectors, device="cuda"), diversity(vectors), atol=1e-6)
