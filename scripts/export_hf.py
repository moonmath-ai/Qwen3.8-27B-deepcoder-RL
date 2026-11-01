"""Export a Zro-form sampler checkpoint as a standard PEFT LoRA adapter directory (for the HF Hub).

The store already holds PEFT-format tensors (get_peft_model_state_dict keys); this writes them as
adapter_model.safetensors next to an adapter_config.json listing only the modules the adapter has.
Run in Zro-form's venv:

  .venv/bin/python scripts/export_hf.py --src <store>/<run>/sampler_weights/final --out <dir> \
      [--note "seed 0, step 100"]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

from safetensors import safe_open

from zroform.trainer import LORA_ALPHA, LoraSpec


def main(a) -> None:
    meta = json.load(open(f"{a.src}/meta.json"))
    with safe_open(f"{a.src}/adapter.safetensors", "pt") as h:
        keys = list(h.keys())
        rank = h.get_slice(next(k for k in keys if k.endswith("lora_A.weight"))).get_shape()[0]
    assert rank == meta["lora_rank"], (rank, meta["lora_rank"])
    present = {k.split(".lora_")[0].rsplit(".", 1)[-1] for k in keys}
    spec = LoraSpec(rank=rank, train_attn=meta["train_attn"], train_mlp=meta["train_mlp"],
                    train_unembed=meta["train_unembed"])
    cfg = spec.peft_config_dict()
    cfg["target_modules"] = [m for m in cfg["target_modules"] if m in present]
    assert set(cfg["target_modules"]) == present, present
    cfg.update(base_model_name_or_path=meta["base_model"], lora_alpha=LORA_ALPHA, inference_mode=True)
    os.makedirs(a.out, exist_ok=True)
    shutil.copyfile(f"{a.src}/adapter.safetensors", f"{a.out}/adapter_model.safetensors")
    json.dump(cfg, open(f"{a.out}/adapter_config.json", "w"), indent=1)
    info = {"base_model": meta["base_model"], "step": meta["step"], "lora_rank": rank,
            "renderer": meta.get("user_metadata", {}).get("renderer_name"), "note": a.note}
    json.dump(info, open(f"{a.out}/training_info.json", "w"), indent=1)
    print(json.dumps({"out": a.out, "tensors": len(keys), **cfg}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    main(ap.parse_args())
