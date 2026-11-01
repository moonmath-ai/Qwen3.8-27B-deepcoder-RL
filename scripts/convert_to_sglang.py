"""Convert this PEFT adapter into the layout SGLang loads for Qwen3.8's linear-attention (Gated DeltaNet) layers.

PEFT saves the fused `in_proj_qkv` projection; SGLang expects separate `in_proj_q`, `in_proj_k`, `in_proj_v`
(it then merges them with `in_proj_z` into its own fused `in_proj_qkvz`). The split is exact: LoRA on a fused
projection equals per-slice LoRA with a shared `lora_A` and `lora_B` split by rows. All other tensors are
copied unchanged. Needs only `torch` and `safetensors`.

  python convert_to_sglang.py <adapter_dir> <output_dir>

<adapter_dir> is a folder with adapter_model.safetensors + adapter_config.json (the repo root, `seed1/`, or
`step50/seed0/`, `step50/seed1/`).
"""

import json
import os
import shutil
import sys

from safetensors.torch import load_file, save_file

# Rows of in_proj_qkv in Qwen3.8-27B: q, k (16 key heads x 128), v (48 value heads x 128)
QKV_SPLIT = [16 * 128, 16 * 128, 48 * 128]


def convert(src: str, dst: str) -> None:
    weights, out = load_file(os.path.join(src, "adapter_model.safetensors")), {}
    for name, t in weights.items():
        if ".in_proj_qkv." not in name:
            out[name] = t
        elif "lora_A" in name:
            for s in "qkv":
                out[name.replace(".in_proj_qkv.", f".in_proj_{s}.")] = t.clone()
        else:
            assert t.shape[0] == sum(QKV_SPLIT), (name, t.shape)
            for s, part in zip("qkv", t.split(QKV_SPLIT, dim=0)):
                out[name.replace(".in_proj_qkv.", f".in_proj_{s}.")] = part.contiguous()
    cfg = json.load(open(os.path.join(src, "adapter_config.json")))
    # SGLang's name for the fused Gated DeltaNet input projection is in_proj_qkvz
    cfg["target_modules"] = sorted({"in_proj_qkvz" if m in ("in_proj_qkv", "in_proj_z") else m
                                    for m in cfg["target_modules"]})
    os.makedirs(dst, exist_ok=True)
    save_file(out, os.path.join(dst, "adapter_model.safetensors"))
    json.dump(cfg, open(os.path.join(dst, "adapter_config.json"), "w"), indent=1)
    if os.path.exists(os.path.join(src, "training_info.json")):
        shutil.copyfile(os.path.join(src, "training_info.json"), os.path.join(dst, "training_info.json"))
    print(f"wrote {dst}: {len(out)} tensors, target_modules {cfg['target_modules']}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    convert(sys.argv[1], sys.argv[2])
