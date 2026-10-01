import copy
import pandas as pd
import pytest
import torch
from mfgds import experiment
from mfgds.checkpoints import Checkpoints, output_lock


CONFIG = {"device": "cpu", "dataset": {"kind": "synthetic", "sizes": [6, 3, 2]},
          "methods": ["random", "feedback", "grip_approx"], "variants": ["full", "no_image"],
          "seeds": [7, 9], "ks": [0, 2], "orderings": ["random", "best_last"],
          "context_budgets": [128], "feedback_candidates": 2, "epochs": 3,
          "index_backend": "sklearn"}


@pytest.mark.parametrize("stage", ["feedback", "training", "evaluation", "complete"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware unavailable"))])
def test_interruption_resume_matches_uninterrupted(tmp_path, monkeypatch, stage, device):
    config = copy.deepcopy(CONFIG)
    config.update(device=device, index_backend="torch")
    expected = experiment.run(config, tmp_path / "reference")
    save = Checkpoints.save
    interrupted = False
    def stop(self, state):
        nonlocal interrupted
        save(self, state)
        if not interrupted and state["progress"]["stage"] == stage:
            interrupted = True
            raise KeyboardInterrupt
    monkeypatch.setattr(Checkpoints, "save", stop)
    with pytest.raises(KeyboardInterrupt):
        experiment.run(config, tmp_path / "resumed")
    monkeypatch.setattr(Checkpoints, "save", save)
    # Fail if any committed prediction is executed again.
    checkpoint = Checkpoints(tmp_path / "resumed").load()
    saved = dict(checkpoint["rows"])
    if stage == "complete":
        def forbidden(*args, **kwargs):
            raise AssertionError("Completed inference repeated")
        monkeypatch.setattr(experiment.MockLMM, "predict", forbidden)
    actual = experiment.run(config, tmp_path / "resumed", resume=True)
    columns = [c for c in expected if c not in {"latency_s", "retrieval_s"}]
    pd.testing.assert_frame_equal(expected[columns], actual[columns])
    for key, row in saved.items():
        assert Checkpoints(tmp_path / "resumed").load()["rows"][key] == row
    for path in (tmp_path / "reference").glob("retriever_*.pt"):
        a = torch.load(path, weights_only=True)
        b = torch.load(tmp_path / "resumed" / path.name, weights_only=True)
        for name in a["state"]:
            assert torch.equal(a["state"][name], b["state"][name])
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "reference" / "training.csv"),
                                  pd.read_csv(tmp_path / "resumed" / "training.csv"))


def test_corrupt_latest_falls_back_and_rejects_mismatch(tmp_path):
    experiment.run(CONFIG, tmp_path)
    manager = Checkpoints(tmp_path)
    state = manager.load()
    newest = manager.root / f"state_{manager.sequence % 2}.pt"
    newest.write_bytes(b"truncated")
    (manager.root / "state_0.pt.tmp").write_bytes(b"incomplete")
    with pytest.warns(UserWarning, match="Ignoring invalid"):
        result = experiment.run(CONFIG, tmp_path, resume=True)
    assert len(result) == len(state["rows"])
    changed = copy.deepcopy(CONFIG)
    changed["epochs"] += 1
    with pytest.raises(ValueError, match="differ"):
        experiment.run(changed, tmp_path, resume=True)
    for path in manager.root.glob("state_*.pt"):
        path.write_bytes(b"broken")
    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="No valid"):
        experiment.run(CONFIG, tmp_path, resume=True)


def test_atomic_write_failure_keeps_previous(tmp_path, monkeypatch):
    from mfgds import checkpoints
    manager = Checkpoints(tmp_path)
    state = {"identity": {}, "rows": {}, "feedback": {}, "training": {}, "zero": {}, "progress": {}}
    manager.save(state)
    def fail(*args):
        raise OSError("simulated power loss before replace")
    monkeypatch.setattr(checkpoints.os, "replace", fail)
    state["progress"] = {"epoch": 2}
    with pytest.raises(OSError):
        manager.save(state)
    assert manager.load()["progress"] == {}


def test_output_lock_releases_after_interruption(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with output_lock(tmp_path):
            with pytest.raises(RuntimeError, match="Another process"):
                with output_lock(tmp_path):
                    pass
            raise KeyboardInterrupt
    with output_lock(tmp_path):
        pass
