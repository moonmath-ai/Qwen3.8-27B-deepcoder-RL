"""Engine: sessions, training runs (adapter slots), ordered execution, sampler hand-off.

Wire rules implemented here (Tinker SDK 0.33.1 wire protocol):
- per model, ops run strictly in seq_id order; early arrivals wait (chunk 1 of a split
  forward_backward is sent last);
- a retried (model_id, seq_id) returns the original request_id instead of running twice.
Tenancy: each run occupies one adapter slot; a run is visible only to the key that created it.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field

from zroform import losses
from zroform.futures import Futures, Result
from zroform.store import LocalStore, parse_path
from zroform.telemetry import MetricsLog
from zroform.trainer import LoraSpec, Trainer
from zroform.wire import (FwdBwdRequest, WireError, encode_forward_backward_output,
                          encode_sample_response)

ADAPTER_VERSIONS_KEPT = 2  # sampler keeps the newest N adapter versions per run loaded


class UserError(ValueError):
    """Bad client input; reported with category=user."""


def owner_of(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]  # never store the key itself


@dataclass
class Run:
    model_id: str
    owner: str
    session_id: str
    base_model: str
    spec: LoraSpec
    slot: str
    user_metadata: dict
    next_seq: int = 1
    seen: dict[int, str] = field(default_factory=dict)            # seq_id -> request_id
    pending: dict[int, tuple] = field(default_factory=dict)       # seq_id -> (request_id, op, args)
    running: bool = False
    adapter_versions: list[str] = field(default_factory=list)
    n_versions: int = 0
    last_request: float = field(default_factory=time.time)
    last_fields: dict = field(default_factory=dict)              # scalars for the metrics log


@dataclass
class SamplingSession:
    owner: str
    base_model: str
    adapter: str | None
    model_path: str | None


class Engine:
    def __init__(self, profile, trainer: Trainer, sampler, store: LocalStore, futures: Futures,
                 metrics: MetricsLog | None = None):
        self.p = profile
        self.trainer = trainer
        self.sampler = sampler
        self.store = store
        self.futures = futures
        self.metrics = metrics or MetricsLog(None)
        self.sessions: dict[str, dict] = {}
        self.runs: dict[str, Run] = {}
        self.sampling: dict[str, SamplingSession] = {}
        self.free_slots = [f"s{i}" for i in range(profile.slots_per_base)]
        self.path_adapters: dict[str, str] = {}   # named sampler checkpoint -> loaded adapter name
        # Adapters currently loaded in the sampler. A sampling session opened from a checkpoint path can
        # outlive the adapter it started with (the training run is released, or the version rotates out):
        # _do_sample then reloads the checkpoint from the store (2026-10-24, eval after training ended).
        self.loaded_adapters: set[str] = set()
        self._reload_lock = asyncio.Lock()

    # ---- sessions ----------------------------------------------------------------------------
    def create_session(self, owner: str, user_metadata: dict | None) -> str:
        sid = f"zf-{uuid.uuid4().hex[:12]}"
        self.sessions[sid] = {"owner": owner, "seen": time.time(), "n_train": 0, "n_sample": 0}
        return sid

    def heartbeat(self, owner: str, sid: str) -> bool:
        s = self.sessions.get(sid)
        if s is None or s["owner"] != owner:
            return False
        s["seen"] = time.time()
        return True

    def _session(self, owner: str, sid: str) -> dict:
        s = self.sessions.get(sid)
        if s is None or s["owner"] != owner:
            raise UserError("unknown session")
        return s

    def run(self, owner: str, model_id: str) -> Run:
        r = self.runs.get(model_id)
        if r is None or r.owner != owner:
            raise UserError("unknown model_id")  # same answer for "missing" and "not yours"
        return r

    # ---- training runs -----------------------------------------------------------------------
    async def create_model(self, owner: str, req: dict) -> str:
        sid = req["session_id"]
        self._session(owner, sid)
        base = req["base_model"]
        if base not in self.p.models or base != self.trainer.base_model:
            raise UserError(f"base_model {base!r} not served here")
        lc = req.get("lora_config") or {}
        spec = LoraSpec(rank=lc.get("rank", 32), train_attn=lc.get("train_attn", True),
                        train_mlp=lc.get("train_mlp", True), train_unembed=lc.get("train_unembed", True))
        if not 1 <= spec.rank <= self.p.models[base].max_lora_rank:
            raise UserError(f"rank must be 1..{self.p.models[base].max_lora_rank}")
        if (req.get("optimizer_config") or {}).get("type", "adamw") != "adamw":
            raise UserError("only the adamw optimizer is supported")
        model_id = f"{sid}:train:{req.get('model_seq_id', 0)}"
        if model_id in self.runs:  # a retried create_model must not take a second slot
            return model_id
        if not self.free_slots:
            raise UserError("no free training slot on this server")
        slot = self.free_slots.pop(0)
        try:  # GPU work off the event loop: the trainer lock may be held by a long forward_backward
            await asyncio.to_thread(self.trainer.create_slot, slot, spec, lc.get("seed"))  # seed is in lora_config
        except Exception:
            self.free_slots.insert(0, slot)
            raise
        self.runs[model_id] = Run(model_id, owner, sid, base, spec, slot, req.get("user_metadata") or {})
        self.metrics.record("create_model", run=model_id, lora_rank=spec.rank, profile_sha256=self.p.sha256)
        return model_id

    async def release(self, model_id: str) -> None:
        """Free a run's slot: zero its adapter/optimizer tensors, drop its sampler adapters."""
        r = self.runs.pop(model_id, None)
        if r is None:
            return
        await asyncio.to_thread(self.trainer.free_slot, r.slot)
        self.metrics.record("release", run=model_id)
        self.free_slots.append(r.slot)
        for name in r.adapter_versions:
            await self._unload(name)
        for path in [p for p, a in self.path_adapters.items() if a in r.adapter_versions]:
            del self.path_adapters[path]

    async def end_session(self, sid: str) -> None:
        """Release a session's runs (GPU state zeroed) and sampling sessions, then forget it."""
        for mid in [m for m, r in self.runs.items() if r.session_id == sid]:
            await self.release(mid)
        for ss in [k for k in self.sampling if k.startswith(sid + ":")]:
            del self.sampling[ss]
        self.sessions.pop(sid, None)

    async def reap(self, idle_sec: float) -> None:
        """End sessions without a heartbeat for idle_sec."""
        now = time.time()
        for sid in [k for k, s in self.sessions.items() if now - s["seen"] > idle_sec]:
            await self.end_session(sid)

    def submit(self, owner: str, model_id: str, seq_id: int, op: str, args) -> str:
        """Queue an op for strict seq_id-ordered execution; dedupe retries."""
        r = self.run(owner, model_id)
        r.last_request = time.time()
        if seq_id in r.seen:
            return r.seen[seq_id]
        if seq_id < r.next_seq:
            raise UserError(f"seq_id {seq_id} already executed")
        rid = self.futures.create()
        r.seen[seq_id] = rid
        r.pending[seq_id] = (rid, op, args)
        if not r.running:
            asyncio.create_task(self._drain(r))
        return rid

    async def _drain(self, r: Run) -> None:
        r.running = True
        try:
            while r.next_seq in r.pending:
                rid, op, args = r.pending.pop(r.next_seq)
                seq, r.next_seq = r.next_seq, r.next_seq + 1
                name = "forward" if op == "forward_backward" and args.forward_only else op
                t0, r.last_fields, failed = time.monotonic(), {}, 1
                try:
                    self.futures.resolve(rid, await getattr(self, f"_op_{op}")(r, args))
                    failed = 0
                except FileNotFoundError:
                    self.futures.fail(rid, "checkpoint not found", "user")
                except (UserError, WireError, losses.LossInputError, ValueError) as e:
                    self.futures.fail(rid, str(e), "user")
                except Exception as e:  # noqa: BLE001 - surfaced to the client as a server error
                    # Type only: messages from deep inside torch/SGLang could echo content.
                    self.futures.fail(rid, f"internal error ({type(e).__name__})", "server")
                fields = {"failed": 1} if failed else r.last_fields
                self.metrics.record(name, run=r.model_id, seq_id=seq, duration_s=time.monotonic() - t0, **fields)
        finally:
            r.running = False

    # ---- ops (run in seq order) --------------------------------------------------------------
    async def _op_forward_backward(self, r: Run, req: FwdBwdRequest) -> Result:
        outs, metrics = await asyncio.to_thread(self.trainer.forward_backward, r.slot, req)
        r.last_fields = {"loss_fn": req.loss_fn, "n_datums": len(req.data),
                         "n_tokens": sum(len(d.tokens) for d in req.data),
                         "loss_sum": metrics.get("loss:sum"), "logprob_gap_mean": metrics.get("logprob_gap:mean"),
                         "n_inert_datums": getattr(self.trainer, "last_inert", 0),
                         **self.trainer.memory_stats()}
        return Result("proto", encode_forward_backward_output(outs, metrics))

    async def _op_optim_step(self, r: Run, adam: dict) -> Result:
        metrics = await asyncio.to_thread(self.trainer.optim_step, r.slot, adam)
        r.last_fields = {"grad_norm": metrics.get("grad_norm"), "step": self.trainer.steps[r.slot],
                         **self.trainer.memory_stats()}
        return Result("json", {"metrics": {f"{k}:mean": v for k, v in metrics.items()}})

    async def _op_save_weights(self, r: Run, a: dict) -> Result:
        name = a.get("path") or f"step-{self.trainer.steps[r.slot]:06d}"
        weights, optim, step = await asyncio.to_thread(self.trainer.state, r.slot)
        path = await asyncio.to_thread(self.store.save, r.model_id, "weights", name, weights, optim,
                                       self._meta(r, step), a.get("ttl_seconds"))
        return Result("json", {"path": path, "type": "save_weights"})

    async def _op_save_weights_for_sampler(self, r: Run, a: dict) -> Result:
        tensors = await asyncio.to_thread(self.trainer.adapter_tensors, r.slot)
        r.n_versions += 1
        adapter = f"{r.slot}-{uuid.uuid4().hex[:8]}-v{r.n_versions}"
        await self._load(adapter, tensors, r.spec.peft_config_dict())
        del tensors
        r.adapter_versions.append(adapter)
        while len(r.adapter_versions) > ADAPTER_VERSIONS_KEPT:
            await self._unload(r.adapter_versions.pop(0))
        if a.get("path"):  # named: also persist sampler weights (adapter only)
            weights, _, step = await asyncio.to_thread(self.trainer.state, r.slot)
            path = await asyncio.to_thread(self.store.save, r.model_id, "sampler_weights", a["path"],
                                           weights, None, self._meta(r, step), a.get("ttl_seconds"))
            self.path_adapters[path] = adapter
            return Result("json", {"path": path})
        ssid = self._new_sampling(r.owner, r.session_id, r.base_model, adapter, None)
        return Result("json", {"sampling_session_id": ssid})

    async def _op_load_weights(self, r: Run, a: dict) -> Result:
        weights, optim, meta = await asyncio.to_thread(self.store.load, a["path"], bool(a.get("optimizer")))
        if meta.get("owner") != r.owner:
            raise UserError("unknown checkpoint")
        await asyncio.to_thread(self.trainer.load_state, r.slot, weights, optim, meta.get("step", 0))
        return Result("json", {"path": a["path"], "type": "load_weights"})

    def _meta(self, r: Run, step: int) -> dict:
        return {"owner": r.owner, "base_model": r.base_model, "step": step, "lora_rank": r.spec.rank,
                "train_attn": r.spec.train_attn, "train_mlp": r.spec.train_mlp,
                "train_unembed": r.spec.train_unembed, "user_metadata": r.user_metadata}

    # ---- sampling ----------------------------------------------------------------------------
    def _new_sampling(self, owner, sid, base, adapter, model_path) -> str:
        s = self.sessions[sid]
        ssid = f"{sid}:sample:{s['n_sample']}"
        s["n_sample"] += 1
        self.sampling[ssid] = SamplingSession(owner, base, adapter, model_path)
        return ssid

    async def create_sampling_session(self, owner: str, req: dict) -> str:
        sid = req["session_id"]
        self._session(owner, sid)
        path = req.get("model_path")
        adapter, base = None, req.get("base_model")
        if path:
            parse_path(path)  # validates the format
            meta = self.store.meta(path)
            if meta.get("owner") != owner:
                raise UserError("unknown checkpoint")
            base = meta["base_model"]
            adapter = self.path_adapters.get(path) or await self._load_from_store(path, meta)
        if base != self.trainer.base_model:
            raise UserError(f"base_model {base!r} not served here")
        return self._new_sampling(owner, sid, base, adapter, path)

    async def _load_from_store(self, path: str, meta: dict) -> str:
        weights, _, _ = await asyncio.to_thread(self.store.load, path, False)
        spec = LoraSpec(meta["lora_rank"], meta["train_attn"], meta["train_mlp"], meta["train_unembed"])
        adapter = f"ckpt-{uuid.uuid4().hex[:12]}"
        await self._load(adapter, weights, spec.peft_config_dict())  # sampler places them
        self.path_adapters[path] = adapter
        return adapter

    def submit_sample(self, owner: str, req: dict) -> tuple[str, list[str]]:
        ssid = req.get("sampling_session_id")
        s = self.sampling.get(ssid)
        if s is None or s.owner != owner:
            raise UserError("unknown sampling_session_id")
        n = int(req.get("num_samples", 1))
        rid = self.futures.create()
        seq_ids = [f"{rid}:{i}" for i in range(n)]
        asyncio.create_task(self._do_sample(rid, s, req, n))
        return rid, seq_ids

    async def _load(self, name: str, tensors: dict, config: dict) -> None:
        await self.sampler.load_adapter(name, tensors, config)
        self.loaded_adapters.add(name)

    async def _unload(self, name: str) -> None:
        self.loaded_adapters.discard(name)
        await self.sampler.unload_adapter(name)

    async def _ensure_adapter(self, s: SamplingSession) -> None:
        """A checkpoint-backed session whose adapter was unloaded gets the checkpoint reloaded from the store."""
        if s.adapter is None or s.adapter in self.loaded_adapters or not s.model_path:
            return
        async with self._reload_lock:  # many concurrent samples from one session: reload once
            if s.adapter in self.loaded_adapters:
                return
            live = self.path_adapters.get(s.model_path)
            s.adapter = live if live in self.loaded_adapters else \
                await self._load_from_store(s.model_path, self.store.meta(s.model_path))

    async def _do_sample(self, rid: str, s: SamplingSession, req: dict, n: int) -> None:
        try:
            await self._ensure_adapter(s)
            if req.get("topk_prompt_logprobs") or req.get("topk_sample_logprobs") or \
                    req.get("prompt_alt_tokens_k") or req.get("target_prompt_logprobs") is not None:
                raise UserError("top-k / alt-token / target prompt logprobs not supported yet")
            toks: list[int] = []
            for c in req["prompt"]["chunks"]:
                if c.get("type", "encoded_text") != "encoded_text":
                    raise UserError("only text prompts are supported")
                toks.extend(c["tokens"])
            t0 = time.monotonic()
            seqs, plp = await self.sampler.sample(toks, n, req.get("sampling_params") or {}, s.adapter,
                                                  bool(req.get("prompt_logprobs")))
            self.futures.resolve(rid, Result("proto", encode_sample_response(seqs, plp)))
            self.metrics.record("sample", duration_s=time.monotonic() - t0, n_samples=n, n_tokens=len(toks),
                                n_output_tokens=sum(len(q.tokens) for q in seqs))
        except UserError as e:
            self.futures.fail(rid, str(e), "user")
        except Exception as e:  # noqa: BLE001
            self.futures.fail(rid, f"sampling failed ({type(e).__name__})", "server")
