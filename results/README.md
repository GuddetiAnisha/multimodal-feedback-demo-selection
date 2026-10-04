# Experiment outputs

This directory contains the committed summary of the completed real ScienceQA Natural Science evaluation plus instructions for generated experiment artifacts.

## Completed ScienceQA Natural Science evaluation

Three seeds (**0, 1, 2**) were completed on CUDA. Each seed produced **42,516 prediction rows**, giving **127,548 combined predictions**.

Best result: **Multimodal demonstration selection at k=2 = 85.224% accuracy**, which is **+2.385 percentage points over random selection** at k=2. Multimodal also led at k=4 with **84.251% accuracy**.

Committed result files:

- `SCIENCEQA_NATURAL_RESULTS.md` — human-readable final results and interpretation.
- `final_method_comparison.csv` — machine-readable final comparison table.

Large generated run directories, checkpoints, prediction CSVs and plots remain excluded from version control because they are produced locally and can be large.

Each local run stores its configuration, data fingerprint, environment versions, split IDs, learned weights, training losses, observed feedback, per-query predictions, statistical summaries, Markdown tables and plots. Existing completed/running result directories are not overwritten; use `--resume` to continue an interrupted run.
