import pytest
from mfgds import checkpoints


def windows_lock(code):
    error = PermissionError("simulated Windows file lock")
    error.winerror = code
    return error


@pytest.mark.parametrize("code", [5, 32, 33])
def test_atomic_replace_retries_windows_lock(tmp_path, monkeypatch, code):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"previous")
    replace = checkpoints.os.replace
    attempts = []
    def locked(source, destination):
        attempts.append(1)
        if len(attempts) < 3:
            assert path.read_bytes() == b"previous"
            raise windows_lock(code)
        replace(source, destination)
    monkeypatch.setattr(checkpoints.os, "replace", locked)
    monkeypatch.setattr(checkpoints.time, "sleep", lambda seconds: None)
    checkpoints.atomic_bytes(path, b"new")
    assert len(attempts) == 3
    assert path.read_bytes() == b"new"
    assert not path.with_name(path.name + ".tmp").exists()


def test_persistent_lock_preserves_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"previous")
    attempts = []
    def locked(*args):
        attempts.append(1)
        raise windows_lock(5)
    monkeypatch.setattr(checkpoints.os, "replace", locked)
    monkeypatch.setattr(checkpoints.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError, match="previous checkpoint is preserved"):
        checkpoints.atomic_bytes(path, b"new")
    assert len(attempts) == 9
    assert path.read_bytes() == b"previous"
    assert path.with_name(path.name + ".tmp").read_bytes() == b"new"


def test_non_lock_error_is_not_retried(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("disk failure")
    def forbidden(*args):
        raise AssertionError("Unexpected retry")
    monkeypatch.setattr(checkpoints.os, "replace", fail)
    monkeypatch.setattr(checkpoints.time, "sleep", forbidden)
    with pytest.raises(OSError, match="disk failure"):
        checkpoints.atomic_bytes(tmp_path / "checkpoint.pt", b"new")
