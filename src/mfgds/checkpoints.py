"""Atomic, checksummed local checkpoints. Only load checkpoints you trust."""
import hashlib
import io
import os
from pathlib import Path
import random
import time
import warnings
from contextlib import contextmanager
import numpy as np
import torch


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Checkpoint CUDA device count differs; use the same GPU setup to resume")
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_bytes(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    # Windows scanners/indexers can briefly open the destination without delete
    # sharing. Never delete the committed generation to work around that lock.
    for attempt in range(9):
        try:
            os.replace(temporary, path)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in {5, 32, 33}:
                raise
            if attempt == 8:
                raise PermissionError(
                    f"Cannot replace {path} after retrying temporary Windows file locks. "
                    "The previous checkpoint is preserved. Close programs inspecting "
                    "this file and check folder write permissions, then run with --resume."
                ) from error
            time.sleep(min(0.1 * 2**attempt, 2.0))


@contextmanager
def output_lock(output):
    """OS lock releases on exit, Ctrl+C or process death; lock file may remain."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another process is using this output directory") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class Checkpoints:
    """Alternate two generations so a corrupt newest file has a fallback."""
    def __init__(self, output):
        self.root = Path(output) / "checkpoints"
        self.root.mkdir(exist_ok=True)
        self.sequence = 0

    def save(self, state):
        state["rng"] = rng_state()
        buffer = io.BytesIO()
        torch.save({"format": 1, "sequence": self.sequence + 1, "state": state}, buffer)
        payload = buffer.getvalue()
        data = hashlib.sha256(payload).hexdigest().encode() + b"\n" + payload
        atomic_bytes(self.root / f"state_{(self.sequence + 1) % 2}.pt", data)
        self.sequence += 1

    def load(self):
        valid = []
        for path in self.root.glob("state_*.pt"):
            try:
                digest, payload = path.read_bytes().split(b"\n", 1)
                if digest.decode() != hashlib.sha256(payload).hexdigest():
                    raise ValueError("checksum mismatch")
                item = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
                if item["format"] != 1 or not isinstance(item["state"], dict):
                    raise ValueError("unsupported checkpoint")
                required = {"identity", "rows", "feedback", "training", "zero", "progress", "rng"}
                if not required <= item["state"].keys():
                    raise ValueError("incomplete checkpoint")
                valid.append(item)
            except Exception as error:
                warnings.warn(f"Ignoring invalid checkpoint {path.name}: {error}")
        if not valid:
            raise RuntimeError("No valid checkpoint; preserve this directory and use a new output for a fresh run")
        latest = max(valid, key=lambda x: x["sequence"])
        self.sequence = latest["sequence"]
        return latest["state"]
