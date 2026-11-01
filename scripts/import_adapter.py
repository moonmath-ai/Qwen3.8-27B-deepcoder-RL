"""Import an external PEFT LoRA adapter (dir or .tar, e.g. SkyRL's export) into Zro-form's store,
so it can be sampled and benchmarked through our server like a run's own checkpoint.

Run in Zro-form's venv; prints the tinker:// path to use with create_sampling_client(model_path=...):
  .venv/bin/python scripts/import_adapter.py \
      --name my_adapter --src <peft_dir_or.tar>
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
from pathlib import Path

from safetensors.torch import load_file

from zroform.config import load_profile
from zroform.engine import owner_of
from zroform.store import LocalStore
from zroform.trainer import ATTN, LINEAR_ATTN, MLP

IMPORT_RUN = "imported"  # pseudo model_id grouping imported adapters in the store


def read_peft(src: Path) -> tuple[dict, dict]:
    if src.is_file() and tarfile.is_tarfile(src):
        with tempfile.TemporaryDirectory() as td, tarfile.open(src) as tf:
            for m in tf.getmembers():
                if Path(m.name).name in ("adapter_config.json", "adapter_model.safetensors"):
                    m.name = Path(m.name).name
                    tf.extract(m, td, filter="data")
            return read_peft(Path(td))
    return json.loads((src / "adapter_config.json").read_text()), load_file(src / "adapter_model.safetensors")


def import_adapter(store: LocalStore, name: str, src: Path, owner: str, base_model: str) -> str:
    cfg, weights = read_peft(src)
    targets = set(cfg["target_modules"])
    allowed = set(ATTN + LINEAR_ATTN + MLP + ["lm_head"])  # LINEAR_ATTN: Qwen3.5/3.8 Gated DeltaNet layers
    if not targets <= allowed:
        raise ValueError(f"unsupported target modules {sorted(targets - allowed)}")
    groups = (set(ATTN) if targets & set(ATTN + LINEAR_ATTN) else set()) | (set(MLP) if targets & set(MLP) else set())
    if not groups <= targets:  # the store records whole groups (attn / mlp / lm_head), as Tinker's LoraConfig does
        raise ValueError(f"partial module groups not supported: {sorted(groups - targets)} missing")
    if cfg.get("lora_alpha", 32) != 32:
        raise ValueError("Zro-form serves adapters with lora_alpha=32 (Tinker convention)")
    meta = {"owner": owner, "base_model": base_model, "step": 0, "lora_rank": cfg["r"],
            "train_attn": bool(targets & set(ATTN + LINEAR_ATTN)), "train_mlp": bool(targets & set(MLP)),
            "train_unembed": "lm_head" in targets, "user_metadata": {"imported_from": str(src)}}
    return store.save(IMPORT_RUN, "sampler_weights", name, weights, None, meta, ttl_seconds=None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--profile", default="profiles/node8x-qwen38-27b-ctx32k.yaml")
    ap.add_argument("--api-key", default=os.environ.get("ZROFORM_API_KEYS", "").split(",")[0])
    a = ap.parse_args()
    prof = load_profile(a.profile)
    print(import_adapter(LocalStore(prof.store_dir), a.name, Path(a.src), owner_of(a.api_key), next(iter(prof.models))))
