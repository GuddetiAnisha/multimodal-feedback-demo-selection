# Run the complete project on your NVIDIA GPU

All included experiment configurations now require CUDA. The ScienceQA pipeline
uses the GPU for CLIP encoding, Qwen inference, utility-network training and
scoring, cosine retrieval, selection similarities and diversity calculations.
Missing CUDA or GPU memory produces an error; there is no automatic CPU offload.
File reading, image decoding, tokenization and saving reports use the CPU as in
normal GPU applications.

## 1. Open the project and install GPU dependencies

Extract the ZIP, open the extracted `multimodal-feedback-demo-selection` folder
in PyCharm, and create/select its `.venv` interpreter (Python 3.10+).

In PyCharm's Terminal, verify that `python` points to that environment:

```powershell
python -c "import sys; print(sys.executable)"
```

Use [PyTorch's official installer](https://pytorch.org/get-started/locally/) and
select Windows / Pip / Python / CUDA. Run its generated command in this terminal.
If you already installed CPU-only PyTorch in this environment, uninstall it first:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
```

Then install the project and confirm that CUDA matrix multiplication works:

```powershell
python -m pip install -e ".[dev,hf]"
python scripts/check_gpu.py
```

The detected system GPU is an NVIDIA RTX 2000 Ada with approximately 16 GiB VRAM.
Its NVIDIA driver is installed, but the Python interpreter inspected during
development has CPU-only PyTorch. Install the CUDA wheel in **your PyCharm
environment**. Installing the CUDA toolkit alone does not replace that wheel.

## 2. Prepare the real dataset

Download ScienceQA separately, following the README's expected folder layout:

```powershell
python scripts/prepare_data.py scienceqa --root data/ScienceQA/data/scienceqa --output data/scienceqa --limit 100
```

The first real run downloads Qwen and CLIP model weights. The ZIP includes the
complete application source, configs, setup instructions and tests; pretrained
weights and benchmark datasets must be downloaded separately.

## 3. Run and resume

```powershell
python run_experiments.py --config configs/scienceqa.yaml --output results/scienceqa_gpu
```

In PyCharm's Run configuration:

- Script: `run_experiments.py`
- Working directory: the extracted project folder
- Interpreter: the project's GPU-enabled `.venv`
- Parameters: `--config configs/scienceqa.yaml --output results/scienceqa_gpu`

Startup must display `Compute device: cuda:0` and the GPU name.
Press Ctrl+C once in the Terminal and wait for the stop message. Later, use:

```powershell
python run_experiments.py --config configs/scienceqa.yaml --output results/scienceqa_gpu --resume
```

For the Run button, add `--resume` to Parameters. Keep the output/checkpoints
folder. Use a new output folder when switching from an old CPU experiment or
changing the configuration or installed Python/PyTorch versions.

An offline trial also requires CUDA but uses synthetic data and a mock model:

```powershell
python run_experiments.py --config configs/mock.yaml --output results/gpu_smoke
python run_experiments.py --config configs/mock.yaml --output results/gpu_smoke --resume
```

The mock trial verifies software; it is not a real-model experiment. The separate
sensor-fusion script also defaults to CUDA. Original experiment grid settings
(including seeds, epochs, k, ordering and budgets) are preserved.
