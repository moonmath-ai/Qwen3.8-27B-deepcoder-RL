"""Linear-attention kernels for the trainer (Qwen3.5/3.8 Gated DeltaNet layers), with a verified fallback.

transformers uses fla's gated-delta-rule kernels and causal-conv1d's conv whenever they import. A broken
install (Triton compile error on a new GPU, ABI mismatch) or a numerics bug would otherwise surface
mid-run, so `select(model, mode)` runs at trainer start:
  - "torch": use transformers' PyTorch implementations.
  - "auto":  if any fast kernel is in use, run one real GDN layer forward+backward with the fast kernels
             and with PyTorch on the same input; on an exception or a mismatch, use PyTorch everywhere.
  - "fast":  the same check, but a failure is an error.
`is_kernel_error` lets the trainer fall back (and retry once) if a fast kernel fails later at run time.
"""

from __future__ import annotations

import copy
import logging
import sys
import traceback

import torch

log = logging.getLogger(__name__)
OUT_TOL, GRAD_TOL = 2e-2, 5e-2  # relative L2 error, bf16 inputs (fast vs PyTorch differ by accumulation order)
_KERNEL_MODULES = ("fla", "causal_conv1d", "triton")


def gdn_layers(model) -> list[torch.nn.Module]:
    return [m for m in model.modules() if type(m).__name__.endswith("GatedDeltaNet")]


def _uses_fast(m) -> bool:
    ns = sys.modules[type(m).__module__]
    return (m.causal_conv1d_fn is not None or m.chunk_gated_delta_rule is not ns.torch_chunk_gated_delta_rule
            or type(m.norm).__module__.startswith("fla"))


def use_torch(m) -> None:
    """Switch one GDN layer to transformers' PyTorch implementations (same weights)."""
    ns = sys.modules[type(m).__module__]
    m.causal_conv1d_fn = None  # the layer then runs F.conv1d
    m.causal_conv1d_update = ns.torch_causal_conv1d_update
    m.chunk_gated_delta_rule = ns.torch_chunk_gated_delta_rule
    m.recurrent_gated_delta_rule = ns.torch_recurrent_gated_delta_rule
    if type(m.norm).__module__.startswith("fla"):
        cls = next(v for k, v in vars(ns).items()  # transformers' own class, not the imported fla one
                   if k.endswith("RMSNormGated") and isinstance(v, type) and v.__module__ == ns.__name__)
        w = m.norm.weight
        norm = cls(m.head_v_dim, eps=m.layer_norm_epsilon).to(device=w.device, dtype=w.dtype)
        norm.weight.data.copy_(w.data)
        m.norm = norm


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def check(layer, device, seq_len: int = 160) -> dict:
    """Fast vs PyTorch on one layer: random bf16 input longer than fla's 64-token chunk, plus padding."""
    ref = copy.deepcopy(layer)
    use_torch(ref)
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(2, seq_len, layer.hidden_size, generator=g).to(device, torch.bfloat16)
    mask = torch.ones(2, seq_len, dtype=torch.long, device=device)
    mask[1, seq_len // 2:] = 0  # right padding, as the trainer's micro-batches have
    outs, grads = [], []
    for m in (layer, ref):
        xi = x.clone().requires_grad_(True)
        y = m(xi, attention_mask=mask)
        y = y[0] if isinstance(y, tuple) else y
        (y.float() * mask[..., None]).pow(2).mean().backward()
        outs.append(y.detach() * mask[..., None])
        grads.append(xi.grad.detach())
    return {"out_rel_err": _rel(outs[0], outs[1]), "grad_rel_err": _rel(grads[0], grads[1])}


def select(model, mode: str = "auto", device: str = "cuda") -> str:
    """Pick kernels for every GDN layer; returns a status for the log ('none' if the model has no GDN)."""
    if mode not in ("auto", "fast", "torch"):
        raise ValueError(f"linear_attn_kernels must be auto, fast or torch, not {mode!r}")
    layers = gdn_layers(model)
    if not layers:
        return "none"
    if mode == "torch" or not any(_uses_fast(m) for m in layers):
        for m in layers:
            use_torch(m)
        return "torch" if mode == "torch" else "torch (fast kernels not installed)"
    if not torch.device(device).type == "cuda":
        for m in layers:
            use_torch(m)
        return "torch (not on a GPU)"
    try:
        r = check(layers[0], device)
        ok = r["out_rel_err"] < OUT_TOL and r["grad_rel_err"] < GRAD_TOL
        reason = f"mismatch {r}" if not ok else ""
    except Exception as e:  # noqa: BLE001 - any kernel failure means fall back
        ok, reason = False, f"{type(e).__name__}: {str(e)[:200]}"
    if ok:
        return f"fast (checked: out {r['out_rel_err']:.1e}, grad {r['grad_rel_err']:.1e})"
    if mode == "fast":
        raise RuntimeError(f"fast linear-attention kernels failed their check: {reason}")
    for m in layers:
        use_torch(m)
    log.warning("linear-attention kernels: falling back to PyTorch (%s)", reason)
    return f"torch (fallback: {reason})"


def is_kernel_error(e: BaseException) -> bool:
    """True if the exception was raised inside fla / causal-conv1d / Triton (not, e.g., an OOM)."""
    if isinstance(e, torch.OutOfMemoryError):
        return False
    files = [f.filename for f in traceback.extract_tb(e.__traceback__)]
    return any(f"/{k}/" in f or f"/{k}_" in f for f in files for k in _KERNEL_MODULES)
