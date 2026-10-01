import json
import numpy as np
import pytest
from mfgds.experiment import run


def test_end_to_end_reproducibility_and_audit(tmp_path):
    config = {"device": "cpu", "dataset": {"kind": "synthetic", "sizes": [9, 6, 3]}, "methods": ["random", "visual", "textual", "multimodal", "feedback", "grip_approx"],
              "variants": ["full"], "seeds": [7], "ks": [0, 2], "orderings": ["best_first"], "context_budgets": [128],
              "feedback_candidates": 3, "epochs": 3}
    a = run(config, tmp_path / "a")
    b = run(config, tmp_path / "b")
    assert len(a) == 36
    assert a.prediction.tolist() == b.prediction.tolist()
    assert a.demo_ids.tolist() == b.demo_ids.tolist()
    assert np.all(a.input_tokens + 8 <= a.context_budget)
    assert a[a.k == 0].effective_k.eq(0).all()
    assert a[a.k == 0].gain.eq(0).all()
    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert manifest["mock"] and manifest["status"] == "complete"
    assert (tmp_path / "a" / "summary.csv").exists()
    assert list((tmp_path / "a").glob("plot_*.png"))
    with pytest.raises(FileExistsError):
        run(config, tmp_path / "a")
