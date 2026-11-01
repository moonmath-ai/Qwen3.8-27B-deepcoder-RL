"""Deployment profile: one YAML file selects every situation-dependent setting and backend.

The engine never branches on where it runs; it reads a profile.
"""

from __future__ import annotations

import hashlib
import os
import string
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelSpec:
    max_lora_rank: int = 64
    # Context cap for the sampler; keeps SGLang's KV reservation sane on a shared GPU.
    max_model_len: int = 8192


@dataclass
class Profile:
    name: str
    host: str = "127.0.0.1"
    port: int = 8100
    # The one SDK version this release speaks; other clients are rejected with a clear error.
    sdk_version: str = "0.33.1"
    # Dev auth: comma-separated keys in this env var.
    api_keys_env: str = "ZROFORM_API_KEYS"
    # Tenancy: 1 = dedicated (one run owns the base model); N > 1 = shared adapter slots.
    slots_per_base: int = 1
    models: dict[str, ModelSpec] = field(default_factory=dict)
    store_dir: str = ""
    # Topology. GPU indices are as this server process sees them at start (after any inherited
    # CUDA_VISIBLE_DEVICES); the server then hides every other GPU (config.restrict_visible_gpus).
    # sampler_gpus empty = SGLang on the trainer's GPU (colocated); N GPUs = SGLang TP=N.
    trainer_device: str = "cuda"
    sampler_gpus: list[int] = field(default_factory=list)
    # SGLang --mem-fraction-static: fraction of GPU memory FREE when SGLang starts (after the trainer
    # and any other users), not of total memory (vLLM-style). Too low and weights alone exceed it.
    sampler_mem_fraction: float = 0.35
    sampler_max_loaded_loras: int = 4
    sampler_port: int = 30100
    # Cap SGLang's concurrent requests at its CUDA-graph range: larger batches run eager, allocate big
    # temporary buffers that SGLang's allocator keeps (79.9 GB seen 2026-10-18), starving the trainer.
    sampler_max_running_requests: int = 256
    # Extra SGLang launch flags for tuning (e.g. ["--cuda-graph-max-bs", "512"]); recorded via the profile hash.
    sampler_extra_args: list[str] = field(default_factory=list)
    # Trainer micro-batching budget (tokens per forward/backward chunk).
    micro_batch_tokens: int = 4096
    # Release the trainer's cached GPU memory after every optim_step. OFF by default: measured 2026-10-18,
    # releasing frees ~10 GB between steps but fragments the allocator, so the peak during forward_backward
    # rose from 19.9 GB to 46-50 GB (worse for a shared GPU). Kept as an option.
    release_gpu_cache: bool = False
    # Linear-attention (Qwen3.5/3.8) trainer kernels: auto = fla/causal-conv1d if they pass a check against
    # PyTorch at start (else PyTorch), fast = same but a failed check is an error, torch = always PyTorch.
    linear_attn_kernels: str = "auto"
    # PYTORCH_CUDA_ALLOC_CONF for the trainer process only (set by zroform.launch before torch loads), e.g.
    # "expandable_segments:True" against fragmentation on long sequences. SGLang never inherits it: its
    # TP custom all-reduce shares CUDA-graph buffers over IPC, which fails with expandable segments.
    trainer_cuda_alloc_conf: str = ""
    # Allow-listed scalar metrics (zroform/telemetry.py). Default: <store_dir>/../metrics/server_<ts>.jsonl
    metrics_dir: str = ""
    # Completed futures are dropped from memory after this many seconds if never retrieved.
    future_ttl_sec: int = 600
    sha256: str = ""  # hash of the profile file, recorded in every run

    def api_keys(self) -> set[str]:
        return {k for k in os.environ.get(self.api_keys_env, "").split(",") if k}


def _expand_dirs(d: dict) -> None:
    """Expand ${VAR} in *_dir values only (machine-local paths, e.g. ${ZROFORM_DATA}/store), so the
    environment can't alter tenancy or isolation settings. An unset variable is an error, not ''."""
    for k, v in d.items():
        if k.endswith("_dir") and isinstance(v, str):
            try:
                d[k] = string.Template(v).substitute(os.environ)
            except KeyError as e:
                raise ValueError(f"profile {k}: environment variable {e} is not set") from None


def restrict_visible_gpus(trainer_device: str, sampler_gpus: list[int]) -> tuple[str, list[int]]:
    """Make only the profile's GPUs visible to this process (and so to SGLang), renumbered from 0.
    Returns the trainer device and sampler GPUs in the new numbering. Call before CUDA initialises.
    Why: PyTorch's first cross-GPU copy (the adapter hand-off) opens a ~0.55 GB CUDA context on
    EVERY visible GPU (measured 2026-10-21, torch 2.13), which a shared box can't afford."""
    t = int(trainer_device.split(":")[1]) if ":" in trainer_device else 0
    gpus = sampler_gpus or [t]
    used = sorted({t, *gpus})
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    phys = visible.split(",") if visible else [str(i) for i in range(max(used) + 1)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(phys[g] for g in used)
    return f"cuda:{used.index(t)}", [used.index(g) for g in gpus]


def load_profile(path: str | os.PathLike) -> Profile:
    raw = Path(path).read_bytes()
    d = yaml.safe_load(raw) or {}
    _expand_dirs(d)
    models = {name: ModelSpec(**(spec or {})) for name, spec in (d.pop("models", {}) or {}).items()}
    prof = Profile(**d, models=models)
    prof.sha256 = hashlib.sha256(raw).hexdigest()
    if prof.slots_per_base < 1:
        raise ValueError("slots_per_base must be >= 1")
    if not prof.models:
        raise ValueError("profile lists no models")
    return prof
