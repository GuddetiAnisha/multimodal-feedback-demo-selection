# Multimodal Feedback-Guided Demonstration Selection

A software-only master's-thesis research prototype for selecting image/question/answer demonstrations for large multimodal models. Includes an offline CPU pipeline and optional Hugging Face integration.

**Synthetic/mock outputs validate software only. They are not benchmark findings.** See [VERIFICATION.md](VERIFICATION.md) for the completed local checks and publication limitations.

## Quick start

Python 3.10+; the original local run used Python 3.11 on Windows. From this repository:

~~~powershell
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m mfgds.experiment --config configs/mock.yaml --output results/my_mock_run
.venv/Scripts/python -m mfgds.experiment --config configs/ablations.yaml --output results/my_ablations
~~~

On Linux/macOS replace .venv/Scripts/python with .venv/bin/python. Mock mode downloads no model or dataset; PyTorch trains the utility network. Each run needs a fresh output directory. Predictions/selections were reproducible in the tested CPU environment; timings vary.

Dependencies and optional extras are in [pyproject.toml](pyproject.toml). [requirements-tested.txt](requirements-tested.txt) records packages from the local verification environment; its CPU PyTorch wheel requires the PyTorch CPU index when reinstalling.

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
configs/           CPU, ablation and real-model configurations
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
python scripts/prepare_data.py scienceqa --root data/ScienceQA/data/scienceqa --output data/scienceqa --limit 100
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

Rearrange or link image directories if the downloaded release uses a different layout. The script partitions official training groups 80%/20% into demonstrations/feedback and holds official validation out for evaluation. The optional limit caps each resulting split; this is not an official full-benchmark evaluation. Pre-register the final split protocol for thesis results.

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

Freeze hyperparameters on a development split, run multiple training seeds and then evaluate held-out benchmark queries. Future extensions include conditional marginal feedback for sets, pairwise/listwise training, longer-text encoders, near-duplicate filtering, embedding/feedback caching, batched inference and resumable expensive sweeps. These are not implemented claims.

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
