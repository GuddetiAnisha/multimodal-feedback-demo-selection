# Resume validation

Validated on Windows with Python 3.14.4 and PyTorch 2.12.1+cpu.

Full suite: **24 passed, 1 skipped**. The skipped test requires optional FAISS.

The resume tests interrupt after committed feedback, training, evaluation and
completion checkpoints, then compare predictions, selected demonstrations,
training histories and exact model tensors against uninterrupted runs across
two seeds and two variants. A completed-run test forbids additional inference.
Additional tests cover damaged newest checkpoints, incomplete temporary files,
both generations corrupted, configuration mismatches, failed atomic replacement,
concurrent-output rejection and lock release after interruption.

Experiment configurations and hyperparameters were preserved. The learned-method
iteration order is now sorted for reproducibility. Checkpoints retain two full
generations; storage and write cost grows with the number of completed units.

These checks use synthetic data and the offline mock model. They establish
software behavior, not ScienceQA accuracy or real-HF/GPU reproducibility. Real
models and benchmark datasets were not downloaded or evaluated. A shutdown can
repeat the work after the most recent valid checkpoint; corrupted-newest fallback
can repeat units committed only in that damaged generation.

Run verification from the installed project:

```powershell
python -m pytest -q
```
