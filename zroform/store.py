"""Checkpoint store behind one interface.

A local directory of plaintext safetensors; another backend (e.g. encrypted or remote) can replace it
without changing callers.
TTLs requested by clients are honoured: expired checkpoints are deleted by `sweep()`.
"""

from __future__ import annotations

import io
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

KINDS = ("weights", "sampler_weights")


def tinker_path(model_id: str, kind: str, name: str) -> str:
    if "/" in name or not name:
        raise ValueError("checkpoint name must be non-empty and contain no '/'")
    return f"tinker://{model_id}/{kind}/{name}"


def parse_path(path: str) -> tuple[str, str, str]:
    if not path.startswith("tinker://"):
        raise ValueError("path must start with tinker://")
    parts = path[len("tinker://"):].split("/")
    if len(parts) != 3 or parts[1] not in KINDS:
        raise ValueError(f"bad tinker path {path!r}")
    return parts[0], parts[1], parts[2]


class LocalStore:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, model_id: str, kind: str, name: str) -> Path:
        return self.root / model_id / kind / name

    def save(self, model_id: str, kind: str, name: str, weights: dict[str, torch.Tensor],
             optim_state: dict | None, meta: dict, ttl_seconds: int | None) -> str:
        d = self._dir(model_id, kind, name)
        d.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in weights.items()}, d / "adapter.safetensors")
        if optim_state is not None:
            buf = io.BytesIO()
            torch.save(optim_state, buf)
            (d / "optim.pt").write_bytes(buf.getvalue())
        meta = {**meta, "expires_at": None if ttl_seconds is None else time.time() + ttl_seconds}
        (d / "meta.json").write_text(json.dumps(meta))
        return tinker_path(model_id, kind, name)

    def load(self, path: str, with_optimizer: bool) -> tuple[dict, dict | None, dict]:
        d = self._dir(*parse_path(path))
        if not (d / "meta.json").exists():
            raise FileNotFoundError(path)
        meta = json.loads((d / "meta.json").read_text())
        optim = None
        if with_optimizer:
            if not (d / "optim.pt").exists():
                raise FileNotFoundError(f"{path} has no optimizer state")
            optim = torch.load(io.BytesIO((d / "optim.pt").read_bytes()), weights_only=True)
        return load_file(d / "adapter.safetensors"), optim, meta

    def meta(self, path: str) -> dict:
        return json.loads((self._dir(*parse_path(path)) / "meta.json").read_text())

    def delete(self, path: str) -> None:
        shutil.rmtree(self._dir(*parse_path(path)), ignore_errors=True)

    def sweep(self) -> int:
        """Delete checkpoints whose TTL has passed. Returns the number removed."""
        n, now = 0, time.time()
        for m in self.root.glob("*/*/*/meta.json"):
            exp = json.loads(m.read_text()).get("expires_at")
            if exp is not None and exp < now:
                shutil.rmtree(m.parent, ignore_errors=True)
                n += 1
        return n
