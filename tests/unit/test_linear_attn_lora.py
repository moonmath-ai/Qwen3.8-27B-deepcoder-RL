"""Qwen3.5 (Gated DeltaNet) LoRA: the trainer adapts the linear-attention projections, and the adapter is
rewritten exactly into the layout SGLang loads (in_proj_qkv -> in_proj_q/k/v with a shared A)."""

import torch

from zroform.sampler import sglang_cli_targets, sglang_target_modules, to_sglang_adapter
from zroform.trainer import LoraSpec, Trainer

PFX = "base_model.model.model.layers.0.linear_attn"


def test_fused_qkv_lora_splits_into_per_slice_lora_with_the_same_update():
    r, d, split = 4, 16, (8, 8, 12)
    A, B = torch.randn(r, d), torch.randn(sum(split), r)
    t = {f"{PFX}.in_proj_qkv.lora_A.weight": A, f"{PFX}.in_proj_qkv.lora_B.weight": B,
         f"{PFX}.in_proj_z.lora_A.weight": torch.randn(r, d), f"{PFX}.in_proj_z.lora_B.weight": torch.randn(12, r),
         f"{PFX}.out_proj.lora_A.weight": torch.randn(r, 12), f"{PFX}.out_proj.lora_B.weight": torch.randn(d, r)}
    out, cfg = to_sglang_adapter(t, {"r": r, "target_modules": ["in_proj_qkv", "q_proj"]}, split)
    rebuilt = torch.cat([out[f"{PFX}.in_proj_{s}.lora_B.weight"] @ out[f"{PFX}.in_proj_{s}.lora_A.weight"]
                         for s in "qkv"])
    assert torch.equal(rebuilt, B @ A)
    assert not any(".in_proj_qkv." in k for k in out) and f"{PFX}.in_proj_z.lora_A.weight" in out
    assert cfg["target_modules"] == ["in_proj_qkvz", "out_proj"]  # from the tensors sent, in SGLang's names


def test_a_qwen3_adapter_passes_through_unchanged():
    t = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 4),
         "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": torch.ones(4, 2)}
    out, cfg = to_sglang_adapter(t, {"target_modules": LoraSpec().target_modules()}, None)
    assert out == t and cfg["target_modules"] == ["down_proj", "q_proj"]
    assert sglang_target_modules(["q_proj", "in_proj_qkv", "in_proj_z", "out_proj"]) == \
        ["q_proj", "in_proj_qkvz", "out_proj"]


def _tiny_qwen35():
    from transformers import AutoModelForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            linear_num_key_heads=2, linear_key_head_dim=16, linear_num_value_heads=4,
                            linear_value_head_dim=16, layer_types=["linear_attention"] * 3 + ["full_attention"])
    return AutoModelForCausalLM.from_config(cfg)


def test_trainer_adapts_qwen35_linear_attention_and_reports_sglang_compatible_targets():
    tr = Trainer("tiny-qwen35", 1, device="cpu", grad_checkpointing=False, model=_tiny_qwen35())
    tr.create_slot("s0", LoraSpec(rank=4))
    adapted = {n.split(".lora_")[0].rsplit(".", 1)[-1] for n, _ in tr.peft.named_parameters() if "lora_A" in n}
    assert {"in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "down_proj"} <= adapted
    assert not adapted & {"in_proj_b", "in_proj_a"}  # SGLang can't serve LoRA there
    assert tr.qkv_split() == (32, 32, 64)
    targets = tr.present_targets(LoraSpec().target_modules())
    assert "in_proj_qkv" in targets and sglang_target_modules(targets).count("in_proj_qkvz") == 1
    # SGLang's CLI can't name in_proj_qkvz/out_proj, so the server is started with `all` (auto-detect).
    assert sglang_cli_targets(targets) == ["all"]
    assert sglang_cli_targets(LoraSpec().target_modules()[:4] + ["gate_proj", "lm_head"]) == \
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "lm_head"]  # Qwen3: unchanged
    out, _ = to_sglang_adapter(tr.adapter_tensors("s0"), LoraSpec(rank=4).peft_config_dict(), tr.qkv_split())
    assert any(".in_proj_v.lora_B" in k for k in out)


# ---- kernel selection / fallback (zroform/kernels.py) -------------------------------------------------
import pytest  # noqa: E402

from zroform import kernels  # noqa: E402


def test_kernel_modes_on_cpu_use_pytorch_and_reject_unknown_modes():
    model = _tiny_qwen35()
    assert kernels.select(model, "auto", "cpu").startswith("torch")
    ns = __import__(type(kernels.gdn_layers(model)[0]).__module__, fromlist=["x"])
    assert all(m.chunk_gated_delta_rule is ns.torch_chunk_gated_delta_rule and m.causal_conv1d_fn is None
               for m in kernels.gdn_layers(model))
    assert kernels.select(model, "torch", "cpu") == "torch"
    with pytest.raises(ValueError):
        kernels.select(model, "fastest", "cpu")


def test_only_errors_raised_inside_kernel_packages_trigger_the_fallback():
    def raise_from(path):
        try:
            exec(compile("raise RuntimeError('triton compile failed')", path, "exec"))
        except RuntimeError as e:
            return e
    assert kernels.is_kernel_error(raise_from("/venv/site-packages/fla/ops/gated_delta_rule/chunk.py"))
    assert kernels.is_kernel_error(raise_from("/venv/site-packages/causal_conv1d/causal_conv1d_interface.py"))
    assert not kernels.is_kernel_error(raise_from("/venv/site-packages/zroform/trainer.py"))
    assert not kernels.is_kernel_error(torch.OutOfMemoryError("CUDA out of memory"))


def test_a_kernel_failure_mid_run_switches_to_pytorch_and_retries_once(monkeypatch):
    tr = Trainer("tiny-qwen35", 1, device="cpu", grad_checkpointing=False, model=_tiny_qwen35())
    tr.kernels = "fast (checked)"
    calls = []

    def flaky(slot, req):
        calls.append(tr.kernels)
        if len(calls) == 1:
            exec(compile("raise RuntimeError('ptxas fatal')", "/venv/site-packages/triton/compiler.py", "exec"))
        return [], {}

    monkeypatch.setattr(tr, "_forward_backward", flaky)
    assert tr.forward_backward("s0", None) == ([], {})
    assert len(calls) == 2 and tr.kernels.startswith("torch (fallback at run time")
    tr.kernels = "fast (checked)"
    monkeypatch.setattr(tr, "_forward_backward", lambda s, r: (_ for _ in ()).throw(torch.OutOfMemoryError("oom")))
    with pytest.raises(torch.OutOfMemoryError):  # not a kernel error: no retry, no switch
        tr.forward_backward("s0", None)
    assert tr.kernels == "fast (checked)"


@pytest.mark.gpu
def test_fast_kernels_match_pytorch_on_a_real_layer():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    model = _tiny_qwen35().to("cuda", torch.bfloat16)
    status = kernels.select(model, "fast", "cuda")  # raises if the check fails
    assert status.startswith("fast"), status
