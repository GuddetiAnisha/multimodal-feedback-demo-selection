"""Run in the same PyCharm interpreter as the experiment."""
import sys
import torch


def main():
    print(f"Python interpreter: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("Install CUDA-enabled PyTorch in this interpreter and check the NVIDIA driver.")
        raise SystemExit(1)
    for index in range(torch.cuda.device_count()):
        info = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {info.name}; VRAM {info.total_memory / 1024**3:.1f} GiB")
        x = torch.ones((16, 16), device=f"cuda:{index}")
        assert (x @ x).sum().item() == 4096
        print("CUDA matrix multiplication passed")


if __name__ == "__main__":
    main()
