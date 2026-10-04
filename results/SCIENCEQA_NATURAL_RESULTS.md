# ScienceQA Natural Science — Final 3-Seed Results

These results were produced with the real ScienceQA Natural Science evaluation workflow on CUDA using three seeds: **0, 1, and 2**.

Each seed produced **42,516 prediction rows**, for **127,548 predictions total** after combining the three completed runs.

## Final method comparison

| Method | k=2 accuracy | k=4 accuracy | Gain vs random (k=2) | Gain vs random (k=4) |
| --- | ---: | ---: | ---: | ---: |
| **Multimodal** | **85.224%** | **84.251%** | **+2.385 pp** | **+1.454 pp** |
| Textual | 84.124% | 83.531% | +1.284 pp | +0.734 pp |
| Feedback | 83.700% | 83.601% | +0.861 pp | +0.804 pp |
| Visual | 83.150% | 82.642% | +0.310 pp | -0.155 pp |
| Random | 82.839% | 82.797% | 0.000 pp | 0.000 pp |
| GRIP Approx | 82.430% | 82.218% | -0.409 pp | -0.579 pp |

At **k=0**, all methods share the same zero-shot accuracy because no demonstrations are selected.

## Main finding

The strongest configuration is **multimodal selection with k=2**, reaching **85.224% accuracy**. This is **2.385 percentage points above random selection** at the same k value.

Multimodal selection also remains the strongest method at **k=4**, with **84.251% accuracy**.

Across the evaluated methods, **k=2 generally performs better than k=4**, indicating that a smaller number of well-selected demonstrations can be more effective than adding more examples.

## Completed seed runs

- Seed 0: 42,516 prediction rows
- Seed 1: 42,516 prediction rows
- Seed 2: 42,516 prediction rows
- Combined total: 127,548 prediction rows

## Reproducibility note

The experiment used separate output folders for each seed so interrupted college-computer sessions could be safely resumed with `--resume`. Completed seed outputs were then combined for the final comparison.

The machine-readable summary is available in [`final_method_comparison.csv`](final_method_comparison.csv).
