# Verification

The original local prototype was verified on Windows / Python 3.11.9 with CPU PyTorch.

- Final local test suite: **12 passed, 1 skipped**, 14.64 seconds.
- Skipped: optional FAISS parity test (FAISS not installed). The sklearn fallback passed.
- Main dry run: **6,480** predictions, three seeds, six methods, five k values, three orderings, two budgets; six plots.
- Ablation smoke run: **2,160** predictions, three seeds, six variants, two methods, five k values; six plots.
- Saved-output audits passed: context budgets including generation reserve, effective k, unique selected IDs, pool membership, split separation, zero-shot consistency, finite scores and feedback reward arithmetic.
- One generated plot was visually inspected and was legible with a prominent mock/synthetic label.
- Hugging Face prompting, input-token accounting and generated-token slicing passed a local test-double contract test. No real model weights were downloaded or evaluated.
- ScienceQA and VQAv2 adapters passed fixture tests. Full benchmark files were not downloaded or evaluated.

All numerical experiment results above describe **synthetic/mock software checks**, not research findings. GRIP-inspired regression is an approximation, not a reproduction.

## GitHub publication

The source was transferred through the GitHub connector from the implementation captured in the development conversation because local filesystem execution was unavailable during publication. Documentation was adjusted for a source-only repository. Generated results, checkpoints, images and local test XML are excluded from Git.

The local verification above predates this transfer. A fresh checkout should rerun the commands below; no post-transfer execution is claimed.

~~~sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m mfgds.experiment --config configs/mock.yaml --output results/dry_run
python -m mfgds.experiment --config configs/ablations.yaml --output results/ablation_smoke
~~~
