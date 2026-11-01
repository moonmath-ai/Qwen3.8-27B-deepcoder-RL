"""Importing an external PEFT adapter into the store."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from zroform.store import LocalStore

spec = importlib.util.spec_from_file_location("imp", Path(__file__).parents[2] / "scripts/import_adapter.py")
imp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imp)


def _peft(d: Path, targets, alpha=32):
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": alpha, "target_modules": targets}))
    save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(4, 8)},
              d / "adapter_model.safetensors")


def test_imports_a_full_attn_mlp_adapter_owned_by_the_importer(tmp_path):
    _peft(tmp_path / "a", imp.ATTN + imp.MLP)
    store = LocalStore(str(tmp_path / "store"))
    path = imp.import_adapter(store, "skyrl", tmp_path / "a", "owner1", "base/model")
    assert path == "tinker://imported/sampler_weights/skyrl"
    w, _, meta = store.load(path, with_optimizer=False)
    assert meta["owner"] == "owner1" and meta["train_attn"] and meta["train_mlp"] and not meta["train_unembed"]
    assert len(w) == 1


@pytest.mark.parametrize("targets,alpha", [(["o_proj", "down_proj"], 32), (imp.ATTN, 16), (["embed_tokens"], 32)])
def test_rejects_adapters_we_cannot_describe_or_serve_faithfully(tmp_path, targets, alpha):
    _peft(tmp_path / "a", targets, alpha)
    with pytest.raises(ValueError):
        imp.import_adapter(LocalStore(str(tmp_path / "store")), "x", tmp_path / "a", "o", "base/model")
