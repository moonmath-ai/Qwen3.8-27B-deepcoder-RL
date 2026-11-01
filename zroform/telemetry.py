"""Server-side metrics log: scalars from an allow-list only.

Each line is one op on one run. Field names AND types are allow-listed; anything else is dropped,
so request content (tokens, text, error messages) cannot reach this file by accident.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path

OPS = {"create_model", "forward_backward", "forward", "optim_step", "save_weights",
       "save_weights_for_sampler", "load_weights", "sample", "release"}
NUMERIC = {
    "seq_id", "duration_s", "n_datums", "n_tokens", "n_samples", "n_output_tokens",
    "loss_sum", "logprob_gap_mean", "grad_norm", "lora_rank", "step", "gpu_mem_gb", "gpu_reserved_gb",
    "n_inert_datums",  # datums with all-zero advantages, run forward-only
    "failed",  # 1 when the op failed; the error text is never logged
}
# Opaque identifiers only: fixed formats, checked by regex, never free text.
IDS = {
    "run": re.compile(r"zf-[0-9a-f]{12}:train:\d+"),
    "profile_sha256": re.compile(r"[0-9a-f]{64}"),
    "loss_fn": re.compile(r"cross_entropy|importance_sampling|ppo|cispo|dro"),
}


def clean(fields: dict) -> dict:
    """Keep allow-listed fields with the right type; drop everything else."""
    out = {}
    for k, v in fields.items():
        if k in NUMERIC and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out[k] = round(v, 6) if isinstance(v, float) else v
        elif k in IDS and isinstance(v, str) and IDS[k].fullmatch(v):
            out[k] = v
    return out


class MetricsLog:
    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, op: str, **fields) -> dict | None:
        if self.path is None or op not in OPS:
            return None
        line = {"ts": round(time.time(), 3), "op": op, **clean(fields)}
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(line) + "\n")
        return line
