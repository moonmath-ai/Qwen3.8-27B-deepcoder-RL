"""LoRA trainer backend: HF transformers + PEFT on one GPU.

One base model, one PEFT adapter per *slot* (1 slot = dedicated, N = shared). Slots never
share a forward pass, and a freed slot's tensors are zeroed before release.
Gradients accumulate across forward_backward calls until optim_step, as in Tinker.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.checkpoint
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM

from zroform import kernels, losses
from zroform.wire import FwdBwdRequest

LORA_ALPHA = 32  # Tinker default (cookbook hyperparam_utils; SkyRL uses the same)
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
# Qwen3.5's Gated DeltaNet (linear-attention) layers. in_proj_b/in_proj_a are left out: SGLang 0.5.21
# can't serve LoRA on them (it fuses them into in_proj_ba, which has no LoRA support).
LINEAR_ATTN = ["in_proj_qkv", "in_proj_z", "out_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
UNEMBED = ["lm_head"]


@dataclass
class LoraSpec:
    rank: int = 32
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = True

    def target_modules(self) -> list[str]:
        """Candidate names; PEFT adapts those the model has (a Qwen3 model has no in_proj_qkv)."""
        return (ATTN + LINEAR_ATTN if self.train_attn else []) + (MLP if self.train_mlp else []) + \
               (UNEMBED if self.train_unembed else [])

    def peft_config_dict(self) -> dict:
        """adapter_config.json content, as SGLang expects for tensor loading."""
        return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": self.rank,
                "lora_alpha": LORA_ALPHA, "lora_dropout": 0.0, "bias": "none",
                "target_modules": self.target_modules()}


LOGPROB_CHUNK = 1024  # positions per chunk when computing target logprobs
# Losses whose loss AND gradient are exactly zero for a datum with all-zero advantages, so such datums
# need only a forward pass (for the logprobs the client expects). Not DRO: its beta*(lp-q)^2 term has a
# gradient even at zero advantage. In GRPO-style RL most groups are uniform (83% on MATH, 2026-10-19).
INERT_SKIP_LOSSES = {"importance_sampling", "ppo", "cispo"}


def _chunk_logprobs(logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    x = logits.float()
    return x.gather(-1, tgt[:, None]).squeeze(-1) - torch.logsumexp(x, dim=-1)


def token_logprobs(logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """log p(tgt[i]) from logits [N, V] without materialising an [N, V] fp32 log-softmax: chunks of
    LOGPROB_CHUNK positions, recomputed in backward (checkpoint), so peak extra memory is one chunk.
    A full fp32 log-softmax OOM'd on long SFT examples (2026-10-19)."""
    parts = []
    for s in range(0, logits.shape[0], LOGPROB_CHUNK):
        sl = slice(s, s + LOGPROB_CHUNK)
        if torch.is_grad_enabled() and logits.requires_grad:
            parts.append(torch.utils.checkpoint.checkpoint(_chunk_logprobs, logits[sl], tgt[sl], use_reentrant=False))
        else:
            parts.append(_chunk_logprobs(logits[sl], tgt[sl]))
    return torch.cat(parts) if parts else logits.new_zeros(0, dtype=torch.float32)


def _chunk_head_logprobs(head: torch.nn.Module, h: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return _chunk_logprobs(head(h), tgt)


def head_logprobs(head: torch.nn.Module, hidden: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """log p(tgt[i]) from final hidden states [N, H] through the (LoRA-wrapped) lm_head, LOGPROB_CHUNK
    positions at a time and recomputed in backward, so the [N, V] logits never exist at once. Full logits
    OOM'd the trainer on one 27k-token coding rollout (13.4 GB at a 248k vocabulary, Qwen3.8-27B, 2026-10-21)."""
    parts = []
    for s in range(0, hidden.shape[0], LOGPROB_CHUNK):
        sl = slice(s, s + LOGPROB_CHUNK)
        if torch.is_grad_enabled():
            parts.append(torch.utils.checkpoint.checkpoint(_chunk_head_logprobs, head, hidden[sl], tgt[sl],
                                                           use_reentrant=False))
        else:
            parts.append(_chunk_head_logprobs(head, hidden[sl], tgt[sl]))
    return torch.cat(parts) if parts else hidden.new_zeros(0, dtype=torch.float32)


class Trainer:
    def __init__(self, base_model: str, max_slots: int, micro_batch_tokens: int = 4096,
                 device: str = "cuda", grad_checkpointing: bool = True, model=None,
                 release_cache: bool = False, skip_inert: bool = True, linear_attn_kernels: str = "auto"):
        self.base_model = base_model
        self.max_slots = max_slots
        self.micro_batch_tokens = micro_batch_tokens
        self.device = device
        self._cuda = torch.device(device).type == "cuda"
        if self._cuda:  # empty_cache, memory stats and stray default-device allocations go to this GPU
            torch.cuda.set_device(torch.device(device))
        # Optionally return cached allocator memory after each optim_step (off by default: it frees ~10 GB
        # between steps but raised the training-time peak from ~20 GB to ~50 GB through fragmentation).
        self.release_cache = release_cache and self._cuda
        self.skip_inert = skip_inert
        self.last_inert = 0  # datums run forward-only in the last forward_backward (for the metrics log)
        # `model` lets CPU tests pass a tiny randomly initialised model.
        self.model = model or AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
        self.model.to(device)
        # Linear leaf names before PEFT wraps any of them (wrapped ones then appear as `.base_layer`).
        self._linear_leaves = {n.rsplit(".", 1)[-1] for n, m in self.model.named_modules()
                               if isinstance(m, torch.nn.Linear)}
        self.model.config.use_cache = False
        # Linear-attention (Qwen3.5) kernels: fast ones only if they pass a check against PyTorch (kernels.py).
        self.kernels = kernels.select(self.model, linear_attn_kernels, device)
        if self.kernels != "none":
            kernels.log.warning("linear-attention kernels: %s", self.kernels)
        if grad_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        self.peft = None
        self.slots: dict[str, LoraSpec] = {}
        self.optims: dict[str, torch.optim.Optimizer] = {}
        self.steps: dict[str, int] = {}
        self._lock = threading.Lock()  # one GPU, one job at a time

    # ---- slots -------------------------------------------------------------------------------
    def create_slot(self, slot: str, spec: LoraSpec, seed: int | None = None) -> None:
        with self._lock:
            if slot in self.slots:
                raise ValueError(f"slot {slot} exists")
            if len(self.slots) >= self.max_slots:
                raise RuntimeError("no free adapter slot on this base model")
            cfg = LoraConfig(**{k: v for k, v in spec.peft_config_dict().items()
                                if k not in ("peft_type",)})
            if seed is not None:
                torch.manual_seed(seed)
            if self.peft is None:
                self.peft = get_peft_model(self.model, cfg, adapter_name=slot)
            else:
                self.peft.add_adapter(slot, cfg)
            params = self._params(slot)
            for p in params:
                p.requires_grad_(True)
            self.slots[slot] = spec
            self.optims[slot] = torch.optim.AdamW(params, lr=0.0, weight_decay=0.0)
            self.steps[slot] = 0

    def free_slot(self, slot: str) -> None:
        with self._lock:
            for p in self._params(slot):
                p.data.zero_()
                p.grad = None
            for st in self.optims.pop(slot).state.values():
                for v in st.values():
                    if torch.is_tensor(v):
                        v.zero_()
            self.peft.delete_adapter(slot)
            del self.slots[slot], self.steps[slot]
            if self._cuda:
                torch.cuda.empty_cache()

    def present_targets(self, names: list[str]) -> list[str]:
        """The subset of LoRA target names this base model actually has (Linear leaf modules)."""
        return [n for n in names if n in self._linear_leaves]

    def qkv_split(self) -> tuple[int, int, int] | None:
        """Row sizes (q, k, v) of a fused linear-attention in_proj_qkv, if the model has one."""
        c = getattr(self.model.config, "text_config", self.model.config)
        if not getattr(c, "linear_num_key_heads", None):
            return None
        k = c.linear_num_key_heads * c.linear_key_head_dim
        return k, k, c.linear_num_value_heads * c.linear_value_head_dim

    def _params(self, slot: str) -> list[torch.nn.Parameter]:
        return [p for n, p in self.peft.named_parameters() if f".{slot}." in n and "lora_" in n]

    # ---- forward / backward ------------------------------------------------------------------
    def _micro_batches(self, data, indices=None):
        """Length-sorted batches whose PADDED size (rows x longest row) stays within micro_batch_tokens.
        A sequence longer than the budget gets a batch of its own."""
        batch, longest = [], 0
        for i in sorted(range(len(data)) if indices is None else indices, key=lambda i: -len(data[i].tokens)):
            n = len(data[i].tokens)
            if batch and (len(batch) + 1) * max(longest, n) > self.micro_batch_tokens:
                yield batch
                batch, longest = [], 0
            batch.append(i)
            longest = max(longest, n)
        if batch:
            yield batch

    def forward_backward(self, slot: str, req: FwdBwdRequest) -> tuple[list[dict], dict]:
        try:
            return self._forward_backward(slot, req)
        except Exception as e:
            # A fast linear-attention kernel failing at run time (e.g. a Triton compile error for a new
            # shape): switch every layer to PyTorch and retry once. Partial gradients are already cleared.
            if not self.kernels.startswith("fast") or not kernels.is_kernel_error(e):
                raise
            for m in kernels.gdn_layers(self.model):
                kernels.use_torch(m)
            self.kernels = f"torch (fallback at run time: {type(e).__name__})"
            kernels.log.warning("linear-attention kernels: %s", self.kernels)
            return self._forward_backward(slot, req)

    def _forward_backward(self, slot: str, req: FwdBwdRequest) -> tuple[list[dict], dict]:
        for d in req.data:
            losses.check_inputs(req.loss_fn, set(d.inputs))
            if len(d.inputs["target_tokens"]) != len(d.tokens) or d.inputs["target_tokens"].ndim != 1:
                raise losses.LossInputError("target_tokens must be 1-D with one entry per input token")
        with self._lock:
            self.peft.set_adapter(slot)
            self.peft.train(not req.forward_only)
            outputs: list[dict | None] = [None] * len(req.data)
            gaps: list = []
            ctx = torch.no_grad() if req.forward_only else torch.enable_grad()
            idx_all = list(range(len(req.data)))
            inert: list[int] = []
            if self.skip_inert and not req.forward_only and req.loss_fn in INERT_SKIP_LOSSES:
                inert = [i for i in idx_all if not np.any(req.data[i].inputs["advantages"])]
            inert_set = set(inert)
            active = [i for i in idx_all if i not in inert_set]
            self.last_inert = len(inert)
            try:
                totals = [0.0]
                self._fwd_bwd_batches(req, ctx, outputs, gaps, totals, active, backward=not req.forward_only)
                if inert:  # exact zero loss/gradient: forward only, for their logprobs
                    self._fwd_bwd_batches(req, torch.no_grad(), outputs, gaps, totals, inert, backward=False)
            except Exception:
                # A failed call must not leave partial gradients for the next optim_step (seen 2026-10-18:
                # an OOM mid-batch, then the client's optim_step applied the half-accumulated gradients).
                for p in self._params(slot):
                    p.grad = None
                if self._cuda:
                    torch.cuda.empty_cache()
                raise
            metrics = {"loss:sum": totals[0]}
            gaps = [g for g in gaps if g is not None]
            if gaps:  # trainer-vs-sampler mismatch, reported every step
                metrics["logprob_gap:mean"] = float(np.mean(gaps))
            return outputs, metrics

    def _fwd_bwd_batches(self, req, ctx, outputs, gaps, totals, indices, backward: bool) -> None:
        """Run micro-batches over `indices`; fills outputs/gaps in place, accumulates the loss in totals[0]."""
        with ctx:
            for idx in self._micro_batches(req.data, indices):
                loss, lps = self._run(idx, req)
                if req.loss_fn != "cross_entropy":
                    gaps += [self._gap(lp, req.data[i]) for i, lp in zip(idx, lps)]
                if backward:
                    loss.backward()
                totals[0] += float(loss.detach())
                for i, lp in zip(idx, lps):
                    outputs[i] = {"logprobs": lp.detach().float().cpu().numpy()}

    @staticmethod
    def _gap(lp: torch.Tensor, d) -> float | None:
        """Mean |trainer logprob - sampler logprob| over sampled positions (cookbook pads prompts with 0)."""
        q = torch.from_numpy(d.inputs["logprobs"].astype(np.float32)).to(lp.device)
        m = q != 0
        return float((lp.detach() - q)[m].abs().mean()) if m.any() else None

    def _run(self, idx: list[int], req: FwdBwdRequest):
        data = [req.data[i] for i in idx]
        T = max(len(d.tokens) for d in data)
        ids = torch.zeros(len(data), T, dtype=torch.long)
        mask = torch.zeros(len(data), T, dtype=torch.long)
        for r, d in enumerate(data):
            ids[r, :len(d.tokens)] = torch.from_numpy(d.tokens)
            mask[r, :len(d.tokens)] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        # Stop at the final hidden states; the LoRA-wrapped lm_head runs per chunk in head_logprobs.
        base = self.peft.get_base_model()
        hidden = base.model(input_ids=ids, attention_mask=mask).last_hidden_state  # [B, T, H]
        loss = torch.zeros((), device=self.device)
        lps = []
        for r, d in enumerate(data):
            n = len(d.tokens)
            tgt = torch.from_numpy(d.inputs["target_tokens"].astype(np.int64)).to(self.device)
            # Position i predicts target_tokens[i] (Tinker: inputs and targets have equal length).
            lp = head_logprobs(base.lm_head, hidden[r, :n], tgt)
            inp = {k: torch.from_numpy(v.astype(np.float32)).to(self.device)
                   for k, v in d.inputs.items() if k != "target_tokens"}
            loss = loss + losses.compute(req.loss_fn, lp, inp, req.config)
            lps.append(lp)
        return loss, lps

    # ---- optimizer ---------------------------------------------------------------------------
    def optim_step(self, slot: str, adam: dict) -> dict:
        with self._lock:
            opt = self.optims[slot]
            for g in opt.param_groups:
                g["lr"] = adam.get("learning_rate", 1e-4)
                g["betas"] = (adam.get("beta1", 0.9), adam.get("beta2", 0.95))
                g["eps"] = adam.get("eps", 1e-12)
                g["weight_decay"] = adam.get("weight_decay", 0.0)
            params = self._params(slot)
            norm = torch.nn.utils.clip_grad_norm_(params, adam.get("grad_clip_norm") or float("inf"))
            opt.step()
            opt.zero_grad(set_to_none=True)
            self.steps[slot] += 1
            if self.release_cache:
                torch.cuda.empty_cache()
            return {"grad_norm": float(norm)}

    def memory_stats(self) -> dict:
        if not self._cuda:
            return {}
        return {"gpu_mem_gb": torch.cuda.memory_allocated(self.device) / 1e9,
                "gpu_reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9}

    # ---- weights -----------------------------------------------------------------------------
    # save_embedding_layers=False: with lm_head as a LoRA target PEFT would otherwise copy the full
    # base embedding/lm_head weights (~0.8 GB for Qwen3-4B) into every adapter export.
    def adapter_tensors(self, slot: str) -> dict[str, torch.Tensor]:
        """PEFT-format adapter weights, still on the GPU (for in-memory sampler sync)."""
        with self._lock:
            sd = get_peft_model_state_dict(self.peft, adapter_name=slot, save_embedding_layers=False)
            return {k: v.detach().clone().contiguous() for k, v in sd.items()}

    def state(self, slot: str) -> tuple[dict[str, torch.Tensor], dict, int]:
        """(adapter weights on CPU, optimizer state_dict, step) for save_state."""
        with self._lock:
            sd = get_peft_model_state_dict(self.peft, adapter_name=slot, save_embedding_layers=False)
            return ({k: v.detach().cpu() for k, v in sd.items()},
                    self.optims[slot].state_dict(), self.steps[slot])

    def load_state(self, slot: str, weights: dict, optim_state: dict | None, step: int) -> None:
        with self._lock:
            set_peft_model_state_dict(self.peft, weights, adapter_name=slot)
            if optim_state is not None:
                self.optims[slot].load_state_dict(optim_state)
            self.steps[slot] = step
