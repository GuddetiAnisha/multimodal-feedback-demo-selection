"""Explicit CPU/CUDA placement shared by training and retrieval."""
import os
import torch


def resolve_device(value="cpu"):
    value = str(value)
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use cpu, auto, cuda or cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Install CUDA-enabled PyTorch in "
                               "your PyCharm interpreter and check the NVIDIA driver. "
                               "Run python scripts/check_gpu.py for diagnostics.")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist")
        # Set before creating tensors/CUDA BLAS operations for repeatable training.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        device = torch.device(f"cuda:{index}")
    return device


def cpu_snapshot(value):
    """Detach saved state from live GPU storage and make checkpoints portable."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return value
