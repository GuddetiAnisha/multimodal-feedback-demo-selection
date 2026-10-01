# Resume validation

Validated on Windows with Python 3.14.4 and PyTorch 2.12.1+cpu.

Latest full suite after leakage-safe dataset preparation: **41 passed, 8 skipped**. One skip requires
optional FAISS; seven skips require CUDA hardware and CUDA-enabled PyTorch.
The machine has an NVIDIA RTX 2000 Ada (16 GiB VRAM, detected with nvidia-smi),
but the verification interpreter has CPU-only PyTorch. Actual GPU execution
remains unverified locally. CUDA tests execute with a CUDA-enabled PyTorch wheel.

The resume tests interrupt after committed feedback, training, evaluation and
completion checkpoints, then compare predictions, selected demonstrations,
training histories and exact model tensors against uninterrupted runs across
two seeds and two variants. A completed-run test forbids additional inference.
Additional tests cover damaged newest checkpoints, incomplete temporary files,
both generations corrupted, configuration mismatches, failed atomic replacement,
concurrent-output rejection and lock release after interruption.

Device tests cover missing-CUDA diagnostics, detached checkpoint snapshots,
PyTorch cosine-index agreement with sklearn, stable ties, device-aware training,
checkpoint restoration, and global placement of the encoder and HF model through
test doubles. The GPU ScienceQA configuration is checked against the original
for identical experiment settings. A CPU smoke run of the updated sensor-fusion
script completed with two training epochs per network.

All shipped experiment configurations now require CUDA and the PyTorch index
backend. The default CLI runs real ScienceQA on CUDA. Tests verify that missing
CUDA causes a clear error instead of CPU fallback. The sensor-fusion entry point
also defaults to CUDA. CPU tests explicitly select CPU for software validation.

Windows checkpoint tests simulate WinError 5/32/33 during atomic replacement,
verify recovery after transient locks, confirm permanent denial preserves the
previous checkpoint, and confirm unrelated I/O errors are not retried. The exact
external program or permissions causing a user's lock cannot be identified from
the traceback alone.

Experiment configurations and hyperparameters were preserved. The learned-method
iteration order is now sorted for reproducibility. Checkpoints retain two full
generations; storage and write cost grows with the number of completed units.

These checks use synthetic data and the offline mock model. They establish
software behavior, not ScienceQA accuracy or real-HF/GPU reproducibility. Real
model weights were not evaluated. The full user-downloaded ScienceQA train and
validation data was read to validate the preparation fix: 12,726 official train
rows, 4,241 validation rows; 1,097 overlapping training rows excluded, 11,629
retained (9,301 demonstrations / 2,328 feedback), all 4,241 validation rows kept.
Strict ID/group/exact-image leakage validation passed. Tests additionally cover
transitive overlap, repeated training images, label-independent grouping and
duplicate IDs. Near-duplicate image detection is not implemented. A shutdown can
repeat the work after the most recent valid checkpoint; corrupted-newest fallback
can repeat units committed only in that damaged generation.

Run verification from the installed project:

```powershell
python -m pytest -q
```
