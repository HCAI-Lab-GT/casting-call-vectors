"""Crash-safe file I/O helpers: writes go to a temp file then os.replace() onto the target."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator


@contextlib.contextmanager
def _atomic_path(path: str | Path, suffix: str = "") -> Iterator[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=suffix + ".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, data: Any, **dump_kwargs: Any) -> None:
    dump_kwargs.setdefault("indent", 2)
    with _atomic_path(path) as tmp:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, **dump_kwargs)


def atomic_write_csv(df: Any, path: str | Path, **to_csv_kwargs: Any) -> None:
    to_csv_kwargs.setdefault("index", False)
    with _atomic_path(path) as tmp:
        df.to_csv(tmp, **to_csv_kwargs)


def atomic_save_safetensors(
    tensors: dict, path: str | Path, metadata: dict[str, str] | None = None
) -> None:
    from safetensors.torch import save_file

    with _atomic_path(path) as tmp:
        save_file(tensors, str(tmp), metadata=metadata)


def is_valid_safetensors(path: str | Path, required_keys: tuple[str, ...] = ()) -> bool:
    """True if the file exists, parses, and contains every key in required_keys."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            keys = set(f.keys())
        return all(k in keys for k in required_keys)
    except Exception:
        return False


def load_json_or_none(path: str | Path) -> Any | None:
    """Return parsed JSON, or None if the file is missing or corrupt (e.g. truncated by a kill)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


@contextlib.contextmanager
def file_lock(path: str | Path) -> Iterator[None]:
    """Exclusive advisory lock on `<path>.lock`, for read-modify-write of shared files across jobs."""
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


PERSONA_VECTOR_KEYS = (
    "prompt_persona_vector",
    "response_persona_vector",
    "all_layers_response_persona_vector",
)
