# Multimodal Feedback-Guided Demonstration Selection

A software-only master's-thesis research prototype for selecting image/question/answer demonstrations for large multimodal models. This edition requires an NVIDIA CUDA GPU by default for model computation. See [START_HERE_PYCHARM.md](START_HERE_PYCHARM.md) for setup.

**Synthetic/mock outputs validate software only. They are not benchmark findings.** See [VERIFICATION.md](VERIFICATION.md) for the completed local checks and publication limitations.

## Quick start

Python 3.10+; the original local run used Python 3.11 on Windows. From this repository:

~~~powershell
python -m venv .venv
# Install CUDA-enabled PyTorch in .venv using the official installer:
# https://pytorch.org/get-started/locally/ (Windows / Pip / Python / CUDA)
.venv/Scripts/python -m pip install -e ".[dev,hf]"
.venv/Scripts/python scripts/check_gpu.py
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m mfgds.experiment --config configs/mock.yaml --output results/my_mock_run
.venv/Scripts/python -m mfgds.experiment --config configs/ablations.yaml --output results/my_ablations
~~~

On Linux/macOS replace .venv/Scripts/python with .venv/bin/python. Mock mode downloads no model or dataset; PyTorch trains the utility network. Start each new run in a fresh output directory; use `--resume` to continue an existing run. Predictions/selections were reproducible in the tested CPU environment; timings vary.

## Stop and resume on the college computer

Every included experiment configuration requires CUDA by default. Follow the GPU/PyCharm setup below first. There is no automatic CPU fallback.

From the project folder in PowerShell, after installing the dependencies above:

~~~powershell
# First session (prepared ScienceQA data and HF dependencies required):
.\.venv\Scripts\python.exe run_experiments.py --config configs/scienceqa.yaml --output results/scienceqa_trial
# Later sessions: use exactly the same config and output directory.
.\.venv\Scripts\python.exe run_experiments.py --config configs/scienceqa.yaml --output results/scienceqa_trial --resume
~~~

For an offline trial, replace `configs/scienceqa.yaml` with `configs/mock.yaml` and use `--output results/my_mock_run`. The existing `python -m mfgds.experiment` and `scripts/run_experiment.py` entry points accept the same options.

Press **Ctrl+C once** in the terminal to stop; wait for the stop message before shutting down. In PyCharm, choose `.venv\Scripts\python.exe` as the interpreter, `run_experiments.py` as the script, and the project folder as the working directory. Set Parameters to `--config configs/scienceqa.yaml --output results/scienceqa_trial`; add `--resume` for subsequent sessions. Prefer Ctrl+C in PyCharm's Terminal. Its Stop button or a sudden shutdown also leaves previously committed checkpoints usable.

Checkpointing occurs after every utility-training epoch, feedback sample, zero-shot query, and evaluation unit (seed, variant, query, method, k, ordering, context budget), plus final completion. Checkpoints include utility-model weights, AdamW state, next-epoch progress, complete loss history, Python/NumPy/PyTorch/CUDA RNG states, selection-generator state and partial results. This training has no scheduler; its state is recorded as `None`. The large HF model remains an inference model and is reloaded from the configured model/revision. Pin that revision for reproducibility.

`--resume` chooses the newest valid checksummed generation and warns if it falls back to the previous generation. Writes are flushed and atomically replaced; incomplete `.tmp` files are ignored. If both generations are invalid or missing, resume fails without silently restarting. The current uncommitted epoch or inference call may be repeated. Committed work is skipped; completed results are rebuilt without repeating inference. Selection draws are replayed cheaply from the seed to preserve the original experiment ordering. CSVs and plots are regenerated from checkpoint results, so an interrupted report can be repaired by resuming.

On Windows, checkpoint replacement retries transient access/sharing errors for
approximately nine seconds. This accommodates temporary locks from file scanners
and other programs without deleting the last committed checkpoint. If access is
still denied, close programs inspecting checkpoint files and verify the output
folder is writable. Keep `results/` and retry the same command with `--resume`.

Keep the **entire output folder**, especially `checkpoints/`, on storage that survives college-PC cleanup. Keep the original config, datasets, model cache and Python environment available. Resume rejects changed configuration, dataset content, external GRIP scores, Python or core package versions. Only one process can write an output folder. Checkpoints contain trusted Python/PyTorch serialization: only resume checkpoints created by your own run. Saving full progress after each unit favors recoverability over disk speed and may be costly for very large sweeps. The separate sensor-fusion extension has its existing behavior; these resume options apply to demonstration-selection experiments.

## NVIDIA GPU in PyCharm

### Natural science only

For the smaller agreed experiment, use `configs/scienceqa_natural_practical.yaml`
with your existing prepared natural-science data:

~~~powershell
.\.venv\Scripts\python.exe run_experiments.py --config configs/scienceqa_natural_practical.yaml --output results/scienceqa_natural_practical_gpu
# Later:
.\.venv\Scripts\python.exe run_experiments.py --config configs/scienceqa_natural_practical.yaml --output results/scienceqa_natural_practical_gpu --resume
~~~

This retains all eligible prepared examples, six methods, CUDA, and 100 utility
training epochs. It uses seed 0, k=0/2/4, best-first ordering, a 4096-token evaluation
budget and four feedback demonstrations per query. The feedback context budget
remains 8192; other model and training settings are unchanged. For the verified
1170 feedback and 2362 evaluation queries, this is 50,728 prediction calls rather
than 1,335,172 in the original natural-science grid (about 26 times fewer).
These counts exclude encoding and token-count preprocessing and do not promise
the same factor of runtime speedup. A single seed cannot estimate between-seed
variability. Stop the larger run before starting this one; its checkpoint/config
cannot be resumed into the smaller grid. Keep its output folder, and use the new
output folder above. Existing natural-science data needs no re-preparation.

Prepare all real natural-science examples with no trial cap. Filtering is applied
to official train and validation before image grouping and leakage exclusions:

~~~powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py scienceqa --root data/ScienceQA/data/scienceqa --output data/scienceqa_natural --subject "natural science"
.\.venv\Scripts\python.exe run_experiments.py --config configs/scienceqa_natural.yaml --output results/scienceqa_natural_gpu
# Later:
.\.venv\Scripts\python.exe run_experiments.py --config configs/scienceqa_natural.yaml --output results/scienceqa_natural_gpu --resume
~~~

Use these parameters in PyCharm's Run configuration. A new data/output directory
preserves the all-subject run; its checkpoints cannot resume the filtered dataset.
The seeds, epochs, methods, k, ordering, context budgets and CUDA requirements are
unchanged. The preparation report records the subject and excluded training IDs.
Results apply to natural science only. The verified downloaded dataset produces
4,690 demonstrations, 1,170 feedback queries and 2,362 validation queries, after
excluding 1,013 training rows overlapping held-out groups. This is a smaller real
experiment, but the unchanged grid still requires many inference calls.

Use the project virtual environment as PyCharm's interpreter and working directory.
Install the NVIDIA driver and **CUDA-enabled PyTorch in that interpreter**. A CUDA
toolkit installation does not turn the CPU-only PyTorch wheel into a GPU wheel.
Use the [official PyTorch installer](https://pytorch.org/get-started/locally/),
select Windows / Pip / Python / the CUDA platform supported by your driver, and
run its generated installation command in PyCharm's Terminal. If you previously
installed CPU-only PyTorch, remove that wheel first:

~~~powershell
python -m pip uninstall -y torch torchvision torchaudio
# Run the CUDA installation command from the official selector here.
python -m pip install -e ".[dev,hf]"
python scripts/check_gpu.py
~~~

The check displays the interpreter, PyTorch CUDA runtime, GPU name and VRAM, and
runs a small CUDA matrix multiplication. Continue only when CUDA is available.
For an offline GPU trial (hash encoder and mock simulator still perform CPU fixture work):

~~~powershell
python run_experiments.py --config configs/mock.yaml --output results/mock_gpu --device cuda
# Stop with Ctrl+C; later:
python run_experiments.py --config configs/mock.yaml --output results/mock_gpu --device cuda --resume
~~~

After preparing ScienceQA as described below, run the full GPU configuration:

~~~powershell
python run_experiments.py --config configs/scienceqa_gpu.yaml --output results/scienceqa_gpu
# Later, keep the same config and output:
python run_experiments.py --config configs/scienceqa_gpu.yaml --output results/scienceqa_gpu --resume
~~~

For PyCharm's Run button: script `run_experiments.py`, working directory the project
folder, Parameters `--config configs/scienceqa_gpu.yaml --output results/scienceqa_gpu`.
Add `--resume` after stopping. Both ScienceQA configs now require CUDA by default;
`python run_experiments.py` also defaults to ScienceQA on GPU. The optional `--device` override controls encoder,
HF model, utility training and retrieval placement without changing the grid or
training hyperparameters. `cuda:N` selects another GPU; `auto` chooses CUDA when
available and otherwise CPU. Explicit `cuda` never silently falls back to CPU.

GPU mode puts CLIP, Qwen, the utility networks, their training tensors, exact cosine
retrieval, selection similarity calculations and diversity calculations on the
GPU. Data decoding, tokenization, random ordering, bookkeeping, checkpoint writes,
CSV reports and plots use CPU. Mock prediction similarity also runs on GPU; the
hash fixture's image decoding and text hashing remain CPU preprocessing.
The startup message and manifest record the compute device and GPU name. The
PyTorch retrieval backend replaces the CPU FAISS/sklearn path in GPU mode and uses
stable ordering for equal scores; no Windows FAISS-GPU installation is needed.

Training checkpoint model/optimizer tensors are detached CPU copies and restored
onto the selected device. CUDA RNG state is restored for resume on the same GPU
setup; deterministic kernels are requested. CPU and GPU floating-point results
can differ. Use a **new output folder** when switching a previous CPU experiment
to GPU or changing the Python/PyTorch installation; existing checkpoint settings
must match. Save hardware/package information with research runs.

Qwen and CLIP share GPU memory. Required VRAM depends on image sizes, number of
demonstrations and context budget; GPU allocation failure does not trigger hidden
CPU offloading. Keep original budgets for comparable experiments. If your GPU
cannot fit them, start a separately named experiment with a deliberately adjusted
config. Small utility networks and tiny retrieval pools can run more slowly on a
GPU due to transfers; this option prioritizes your requested placement.

The separate sensor-fusion script also supports GPU training and evaluation:

~~~powershell
python scripts/run_sensor_fusion_robustness.py --device cuda --output results/sensor_gpu.json
~~~

That script's default device is `cuda`; it does not use the demonstration-selection
resume mechanism.

Dependencies and optional extras are in [pyproject.toml](pyproject.toml). [requirements-tested.txt](requirements-tested.txt) records the historical CPU verification environment and must not be used to install this GPU edition.

## Layout

~~~text
src/mfgds/
  data.py          Examples, group-safe partitioning, benchmark adapters
  embeddings.py    Image/question/answer encoders and retrieval views
  retrievers.py    FAISS/sklearn search, PyTorch utility, external GRIP scores
  selectors.py     Diversity selection, ordering and context limits
  models.py        Mock simulator and Hugging Face multimodal chat
  evaluation.py    Scores, clustered confidence intervals, plots and tables
  experiment.py    Feedback collection, training and experiment grid
configs/           GPU-required mock, ablation and real-model configurations
scripts/           Run experiments, prepare data, aggregate seed runs
tests/             Leakage, retrieval, learning, adapters and integration tests
results/           Generated experiments, excluded from Git
~~~

The namespaced layout avoids collisions with Python's selectors module.

## Pipeline and leakage controls

1. Load disjoint demonstration, feedback-query and evaluation-query splits. Reject overlapping IDs, task/image groups and exact image content. VQAv2 questions sharing an image stay together. Near-duplicate detection is not implemented.
2. Represent each demonstration as concatenated normalized image, question/instruction-plus-choices and answer channels. Queries have a zero answer channel. Example.query() removes canonical and annotator answers before retrieval or inference. ScienceQA lectures/solutions are excluded.
3. Sample feedback demonstrations uniformly without replacement. Measure reward(q,d) = score(model(q,[d])) - score(model(q,[])) on feedback queries only. Retain positive, zero and negative rewards. Reuse sampled feedback across ablations within a seed.
4. Train a PyTorch MLP on [q, d, q*d, abs(q-d)] using MSE against observed score changes. A tanh output bounds utility to [-1,1]. Training is full-batch with fixed epochs; there is no early stopping or evaluation-label tuning.
5. Retrieve an exact cosine shortlist, rerank by predicted utility, and greedily select utility minus lambda times maximum positive similarity to an already selected demonstration.
6. Order the examples, then remove whole examples from the end until the query plus generation reserve fits the context budget. Log retained IDs and effective k.
7. Evaluate the fixed retriever on held-out queries. Targets are read only by evaluation scoring.

The diversity rule is a heuristic; single-demo gains do not capture arbitrary set interactions. Ordering and truncation can interact. Use a large enough context budget when isolating ordering effects. Use a separate development set for hyperparameter tuning.

## Encoders and retrieval methods

The hash encoder uses deterministic signed word hashing and downsampled RGB pixels. It is an offline fixture, not a pretrained semantic encoder. CLIP mode uses sentence-transformers clip-ViT-B-32 to encode PIL images and text in a shared space. Missing images are zero vectors. CLIP's text length limit can truncate long questions/answers; long-text encoder alternatives are a research extension.

| Method | Ranking |
| --- | --- |
| random | Seeded uniform pool sampling with random scores |
| visual | Image cosine similarity |
| textual | Query question against demonstration question-plus-answer |
| multimodal | Fused image/question/answer cosine similarity |
| feedback | Multimodal shortlist, then learned full-demonstration utility |
| grip_approx | Visual shortlist, then vision-only utility regression |
| grip_external | Scores exported by an independently run GRIP implementation |

Textual and multimodal retrieval sum their normalized channels; the learned retriever keeps full concatenated channels. Cross-modal sums in hash mode have no semantic interpretation.

Random/similarity baselines do not use diversity penalties. Learned methods use the configured penalty. FAISS IndexFlatIP is used when installed, with exact sklearn cosine search as the fallback. Set index_backend: sklearn to force the fallback. This is exact retrieval, not approximate ANN. The candidate_pool cap limits reranking cost and can exclude useful demonstrations.

Encoding is eager; feedback and model calls are serial. Start with controlled subsets before scaling.

## Ablations and experiment grids

[configs/mock.yaml](configs/mock.yaml) covers three seeds, k=0/1/2/4/8, best-first/best-last/random ordering, and two context budgets.

[configs/ablations.yaml](configs/ablations.yaml) retrains six variants:

- full: all representation channels, feedback and diversity.
- no_image, no_question, no_answer: zero the respective representation channel during training and retrieval.
- no_diversity: remove the redundancy penalty.
- no_feedback: substitute multimodal similarity for the feedback method.

These are retriever-representation ablations; model prompts retain the complete demonstrations.

## Hugging Face model integration

~~~sh
python -m pip install -e ".[hf]"
# Optional, platform dependent:
python -m pip install -e ".[faiss]"
python scripts/prepare_data.py scienceqa --root data/ScienceQA/data/scienceqa --output data/scienceqa
python -m mfgds.experiment --config configs/scienceqa.yaml --output results/scienceqa_trial
~~~

Download the benchmark separately and check its terms. The adapter expects:

~~~text
data/ScienceQA/data/scienceqa/
  problems.json
  pid_splits.json
  images/train/<problem_id>/image.png
  images/val/<problem_id>/image.png
~~~

Rearrange or link image directories if the downloaded release uses a different layout. The script holds official validation intact for evaluation, excludes training components sharing exact image bytes or original task groups with validation, and partitions the remaining connected training groups 80%/20% into demonstrations/feedback. Repeated training images are kept together. The default applies no sample cap. Counts and excluded training IDs are written to `preparation_report.json`; official source data remains unchanged. Optional `--limit` caps each resulting split. This protocol evaluates official validation, not the official test benchmark. Pre-register the split protocol for thesis results.

The model adapter uses AutoProcessor, AutoModelForImageTextToText, interleaved image/question and assistant-answer chat messages, and deterministic generation. It targets decoder-only, multi-image chat checkpoints such as Qwen2.5-VL. Not every checkpoint supports this format. Pin revision to a model commit and run a real-model smoke test on your hardware before a sweep.

Memory and supported context length depend on model, image resolution and number of demonstrations. Set context budgets within the model's supported limits. Compatible real models may run on CPU slowly.

Token budgets use actual processor input IDs, including image placeholders, plus max_new_tokens. Mock counts are labeled estimates. Context fitting repeats preprocessing; it does not silently truncate partial demonstrations. The LMM is not fine-tuned.

**Real model weights and benchmark datasets were not evaluated in the CPU verification.** The Hugging Face adapter has a local test-double contract test. See the [official multimodal chat documentation](https://huggingface.co/docs/transformers/v4.57.1/chat_templating_multimodal).

## VQAv2 and custom JSONL

Arrange official VQAv2 files under one root:

~~~text
v2_OpenEnded_mscoco_train2014_questions.json
v2_mscoco_train2014_annotations.json
v2_OpenEnded_mscoco_val2014_questions.json
v2_mscoco_val2014_annotations.json
train2014/COCO_train2014_000000000009.jpg
val2014/...
~~~

~~~sh
python scripts/prepare_data.py vqav2 --root data/vqav2_raw --output data/vqav2 --limit 100
~~~

Copy the ScienceQA config and change the three dataset paths to the VQAv2 JSONL outputs. All annotator answers are retained only for scoring. This adapter uses labeled train/validation data, not blind test-server submission.

Canonical JSONL, one object per line:

~~~json
{"id":"sample-1","question":"What color is the object?","answer":"red","image":"images/1.jpg","choices":[],"answers":[],"group":"image-1"}
~~~

Custom image paths resolve relative to the JSONL file. Adapter-generated paths are absolute and must be regenerated when moving data. Multiple-choice targets use uppercase letters A, B, etc. The answers array holds VQA annotator responses; group identifies shared image/task units. Splits must be nonempty and labeled. Text-only examples use image: null.

## GRIP comparison boundary

The [GRIP paper](https://arxiv.org/abs/2606.12744) describes vision-only feedback-guided contrastive retrieval. **grip_approx is not a reproduction:** it regresses score differences with an MLP rather than reproducing the paper's contrastive loss, sampling, training protocol or results. Its diversity selection uses visual features. Report it as “vision-only utility regression (GRIP-inspired).”

For an independently implemented GRIP model, export:

~~~json
{"query-id": {"demo-id-1": 0.73, "demo-id-2": -0.12}}
~~~

Set grip_scores to that JSON path and include grip_external in methods. Larger scores rank higher; missing/nonfinite values raise errors. Scores are consumed on a multimodal shortlist; set candidate_pool to the full demonstration pool for full-ranking comparison. Set the redundancy penalty to zero for a pure external ranking comparison.

The external producer must not use evaluation labels for training. Record its code revision, data, checkpoints and configuration separately. This interface cannot audit its training provenance.

## Outputs, metrics and statistics

Each run writes:

| File | Contents |
| --- | --- |
| manifest.json | Completion status, config, split IDs, data hash, versions and backend |
| predictions.csv | Predictions, selected IDs, scores, zero-shot gains, effective k, tokens and timing |
| feedback.csv | Observed single-demo training feedback |
| feedback_summary.csv | Reward means and positive/negative fractions per seed |
| training.csv | Per-epoch MSE |
| retriever_*.pt | Reloadable PyTorch state dictionaries |
| summary.csv, table.md | Aggregated metrics and intervals |
| plot_*.png | Plots separated by variant, ordering and budget |

Evaluation gain measures usefulness of the whole selected set. predicted_utility is the average learned single-demo utility, not measured causal attribution to each selected example.

ScienceQA scoring accepts strict option letters or normalized exact option text. General QA uses normalized exact match. VQA uses leave-one-annotator-out consensus with **simplified normalization**, not a drop-in replacement for the [official VQA evaluator](https://visualqa.org/evaluation.html). Use the official evaluator for publishable VQAv2 results, especially for punctuation/contractions.

Confidence intervals average seed repetitions per query, bootstrap query groups and weight group sums by query counts. Gain intervals are paired to each query's zero-shot score. They condition on fixed training data/models, exclude full retraining/data-split uncertainty, and do not correct for multiple comparisons. A single seed's reported standard deviation is zero by convention, not evidence of certainty.

latency_s covers prediction preprocessing/generation. retrieval_s covers candidate retrieval/reranking, excluding later selection/context fitting. Both exclude one-time loading, embeddings and training. Diversity is mean pairwise 1-cosine; redundancy is mean positive cosine in the variant's full representation. Both are zero for fewer than two examples. These are diagnostics, not quality guarantees.

~~~sh
python scripts/aggregate.py results/run_seed0 results/run_seed1 --output results/combined
~~~

Aggregation rejects duplicate observations and incompatible data/configuration/dependency manifests. Generated results/checkpoints are intentionally excluded from Git. No experimental accuracy claim is made here.

## Research use and extensions

Begin with zero-shot and similarity baselines on a small subset. Inspect the reward distribution: accuracy-difference feedback may be sparse or identically zero. Increasing feedback coverage or adding a separately validated probabilistic reward may help. A decreasing training loss alone does not prove useful retrieval.

Freeze hyperparameters on a development split, run multiple training seeds and then evaluate held-out benchmark queries. Future extensions include conditional marginal feedback for sets, pairwise/listwise training, longer-text encoders, near-duplicate filtering and batched inference. These are not implemented claims.

References: [ScienceQA](https://github.com/lupantech/ScienceQA), [VQA evaluation](https://visualqa.org/evaluation.html), [GRIP](https://arxiv.org/abs/2606.12744), [Sentence Transformers image/text models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#image-text-models).


## Multimodal Sensor Fusion Robustness Extension

This repository now also includes a lightweight software-only prototype for
studying missing-modality robustness in a multimodal perception pipeline.

The extension is designed to support research questions related to robust
autonomous-driving perception without claiming a full vehicle sensor stack.

Implemented components:

- synthetic camera-, LiDAR-, and RADAR-like feature generation from a shared
  latent scene representation;
- feature-level multimodal fusion using a PyTorch classifier;
- controlled modality dropout for camera, LiDAR, and RADAR channels;
- learned missing-modality completion using the remaining modalities;
- graceful-degradation evaluation with accuracy and macro-F1;
- deterministic seeds, automated tests, and JSON experiment reports.

Run the experiment with:

~~~sh
python scripts/run_sensor_fusion_robustness.py --output results/sensor_fusion_robustness.json
~~~

The experiment compares the full-modality baseline with each single-modality
dropout condition and with completion-assisted recovery.

**Scope limitation:** this extension uses synthetic feature vectors. It does
not use real camera/LiDAR/RADAR measurements, bird's-eye-view geometry,
3D bounding boxes, a pretrained BEV detector, or physically realistic adverse
weather. It should be described as a multimodal sensor-fusion robustness
prototype, not as a production autonomous-driving perception system.
