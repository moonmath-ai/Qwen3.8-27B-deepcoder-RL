"""Tinker built-in losses, per datum, token-level and summed (tinker-docs .../tinker/losses/*).

Each function takes the trainer's logprobs of the target tokens (requires grad) plus the
datum's loss_fn_inputs, and returns a scalar loss to add up. All losses are sums, never means.
"""

from __future__ import annotations

import torch

# Tinker rejects any extra key for the RL losses (e.g. `weights`), so do we.
RL_KEYS = {"target_tokens", "logprobs", "advantages"}
CE_KEYS = {"target_tokens", "weights"}

# Defaults not stated as values in Tinker's docs (assumptions, revisit if behaviour differs):
# - ppo: docs say epsilon is fixed at 0.2, so ratio bounds 0.8/1.2.
# - dro: docs show beta=0.05 only as an example.
DEFAULTS = {
    "ppo": {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
    "cispo": {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0},  # documented defaults
    "dro": {"beta": 0.05},
}


class LossInputError(ValueError):
    pass


def check_inputs(loss_fn: str, keys: set[str]) -> None:
    if loss_fn == "cross_entropy":
        if "target_tokens" not in keys or not keys <= CE_KEYS:
            raise LossInputError(f"cross_entropy takes {sorted(CE_KEYS)}, got {sorted(keys)}")
    elif loss_fn in ("importance_sampling", "ppo", "cispo", "dro"):
        if keys != RL_KEYS:
            raise LossInputError(f"{loss_fn} takes exactly {sorted(RL_KEYS)}, got {sorted(keys)}")
    else:
        raise LossInputError(f"unknown loss_fn {loss_fn!r}")


def compute(loss_fn: str, lp: torch.Tensor, inp: dict[str, torch.Tensor], config: dict) -> torch.Tensor:
    """lp: float32 [N] target logprobs under the current policy. Returns a scalar loss."""
    cfg = {**DEFAULTS.get(loss_fn, {}), **(config or {})}
    if loss_fn == "cross_entropy":
        w = inp.get("weights")
        w = torch.ones_like(lp) if w is None else w
        return (-lp * w).sum()
    q = inp["logprobs"]          # sampler logprobs
    adv = inp["advantages"]
    ratio = torch.exp(lp - q)
    if loss_fn == "importance_sampling":
        return -(ratio * adv).sum()
    if loss_fn == "ppo":
        clipped = torch.clamp(ratio, cfg["clip_low_threshold"], cfg["clip_high_threshold"])
        return -torch.min(ratio * adv, clipped * adv).sum()
    if loss_fn == "cispo":
        clipped = torch.clamp(ratio, cfg["clip_low_threshold"], cfg["clip_high_threshold"])
        return -(clipped.detach() * lp * adv).sum()
    if loss_fn == "dro":
        return -(lp * adv - 0.5 * cfg["beta"] * (lp - q) ** 2).sum()
    raise LossInputError(f"unknown loss_fn {loss_fn!r}")
