"""Tinker-compatible HTTP API (pinned SDK 0.33.1).

Run: ZROFORM_PROFILE=profiles/node8x-qwen38-27b-ctx32k.yaml ZROFORM_API_KEYS=tml-... \
       python -m zroform.launch --host 127.0.0.1 --port 8100    (restricts the visible GPUs first)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from zroform.config import load_profile
from zroform.engine import Engine, UserError, owner_of
from zroform.futures import Futures, Result
from zroform.store import LocalStore
from zroform.telemetry import MetricsLog
from zroform.wire import WireError, decode_forward_backward

log = logging.getLogger("zroform")
LONG_POLL_SEC = 30          # client per-poll timeout is 45 s
SESSION_IDLE_SEC = 60       # heartbeats arrive every 10 s; the cookbook never closes sessions, so a
                            # finished client must not hold a (dedicated) slot for long


def _err(status: int, msg: str) -> JSONResponse:
    # Always a non-empty body: an empty 400 makes the SDK retry it as a Cloudflare reject.
    return JSONResponse({"error": msg, "category": "user"}, status_code=status)


def build_engine(profile, trainer=None, sampler=None) -> Engine:
    """Real backends unless the caller (tests) passes its own."""
    base = next(iter(profile.models))
    spec = profile.models[base]
    from zroform.trainer import LoraSpec, Trainer  # heavy imports only for real backends
    if trainer is None:
        trainer = Trainer(base, profile.slots_per_base, profile.micro_batch_tokens,
                          device=profile.trainer_device, release_cache=profile.release_gpu_cache,
                          linear_attn_kernels=profile.linear_attn_kernels)
    if sampler is None:
        import torch
        from zroform.sampler import SGLangSampler
        # Empty = colocated on the trainer's GPU (the Trainer has made it the current device).
        gpus = profile.sampler_gpus or [torch.cuda.current_device()]
        # SGLang needs every module an adapter may target, and only modules this model has.
        targets = trainer.present_targets(LoraSpec().target_modules())
        sampler = SGLangSampler(base, profile.sampler_port, spec.max_lora_rank, targets,
                                profile.sampler_max_loaded_loras, profile.sampler_mem_fraction,
                                spec.max_model_len, str(Path(profile.store_dir).parent / "sglang.log"),
                                profile.sampler_max_running_requests, profile.sampler_extra_args, gpus,
                                trainer.qkv_split())
    mdir = Path(profile.metrics_dir or Path(profile.store_dir).parent / "metrics")
    metrics = MetricsLog(str(mdir / f"server_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"))
    return Engine(profile, trainer, sampler, LocalStore(profile.store_dir), Futures(profile.future_ttl_sec),
                  metrics)


def create_app(profile, engine: Engine | None = None) -> FastAPI:
    state: dict = {}

    @contextlib.asynccontextmanager
    async def lifespan(app):
        eng = engine or build_engine(profile)
        await eng.sampler.start()
        state["engine"] = eng

        async def housekeeping():
            while True:
                await asyncio.sleep(10)
                await eng.reap(SESSION_IDLE_SEC)
                await asyncio.to_thread(eng.store.sweep)
        task = asyncio.create_task(housekeeping())
        log.info("zroform up: profile=%s sha256=%s", profile.name, profile.sha256[:12])
        yield
        task.cancel()
        await eng.sampler.stop()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def gate(request: Request, call_next):
        if request.url.path == "/api/v1/healthz":
            return await call_next(request)
        # Version gate: one pinned SDK version per release.
        ver = request.headers.get("x-stainless-package-version")
        if ver != profile.sdk_version:
            return _err(400, f"unsupported SDK version {ver!r}: install tinker=={profile.sdk_version}")
        key = request.headers.get("x-api-key", "")
        if key not in profile.api_keys():
            return _err(401, "invalid API key")
        request.state.owner = owner_of(key)
        return await call_next(request)

    @app.exception_handler(KeyError)
    async def missing_field(request, e):
        return _err(400, f"missing field {e}")

    @app.exception_handler(FileNotFoundError)
    async def not_found(request, e):
        return _err(400, "checkpoint not found")  # never echo server filesystem paths

    for exc in (UserError, WireError, ValueError):
        app.add_exception_handler(exc, lambda request, e: _err(400, str(e) or type(e).__name__))

    def eng() -> Engine:
        return state["engine"]

    def promise(rid: str, **extra) -> dict:
        return {"request_id": rid, **extra}

    def submit_fn(o, fn, *args) -> str:
        """Run a non-ordered async op (create_model) behind a future."""
        rid = eng().futures.create()

        async def go():
            try:
                eng().futures.resolve(rid, Result("json", await fn(o, *args)))
            except UserError as e:
                eng().futures.fail(rid, str(e), "user")
            except Exception as e:  # noqa: BLE001
                eng().futures.fail(rid, f"internal error ({type(e).__name__})", "server")
        asyncio.create_task(go())
        return rid

    # ---- bootstrap ---------------------------------------------------------------------------
    @app.get("/api/v1/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/api/v1/client/config")
    async def client_config():
        return {"use_pyqwest_transport": False}

    @app.post("/api/v1/client/dynamic_config")
    async def dynamic_config():
        return {}

    @app.post("/api/v1/create_session")
    async def create_session(request: Request):
        body = await request.json()
        return {"session_id": eng().create_session(request.state.owner, body.get("user_metadata")),
                "type": "create_session"}

    @app.post("/api/v1/session_heartbeat")
    async def heartbeat(request: Request):
        body = await request.json()
        if not eng().heartbeat(request.state.owner, body.get("session_id")):
            return JSONResponse({"error": "session expired", "category": "user"}, status_code=410)
        return {"type": "session_heartbeat"}

    @app.post("/api/v1/telemetry")
    async def telemetry():
        # Sink: client telemetry can carry request headers incl. X-API-Key. Never stored or logged.
        return {"status": "accepted"}

    @app.post("/api/v1/sessions/{sid}/finish")
    async def finish(sid: str, request: Request):
        if eng().heartbeat(request.state.owner, sid):  # owner check
            await eng().end_session(sid)               # free its slot now (ServiceClient.close)
        return {}

    @app.get("/api/v1/get_server_capabilities")
    async def capabilities():
        return {"supported_models": [{"model_name": m, "max_context_length": s.max_model_len,
                                      "trainable": True, "sampleable": True}
                                     for m, s in profile.models.items()]}

    # ---- training ----------------------------------------------------------------------------
    @app.post("/api/v1/create_model")
    async def create_model(request: Request):
        body = await request.json()
        async def create(o, b):
            return {"model_id": await eng().create_model(o, b), "type": "create_model"}
        rid = submit_fn(request.state.owner, create, body)
        return promise(rid)

    @app.post("/api/v1/forward_backward")
    async def forward_backward(request: Request):
        req = decode_forward_backward(await request.body(), request.headers.get("content-encoding"))
        rid = eng().submit(request.state.owner, req.model_id, req.seq_id, "forward_backward", req)
        return promise(rid, model_id=req.model_id)

    @app.post("/api/v1/optim_step")
    async def optim_step(request: Request):
        b = await request.json()
        adam = b.get("adam_params") or b.get("optimizer_params") or b.get("optim_params") or {}
        if adam.get("type", "adam") not in ("adam", "adamw"):
            raise UserError("only Adam parameters are supported")
        rid = eng().submit(request.state.owner, b["model_id"], b["seq_id"], "optim_step", adam)
        return promise(rid, model_id=b["model_id"])

    @app.post("/api/v1/save_weights")
    async def save_weights(request: Request):
        b = await request.json()
        rid = eng().submit(request.state.owner, b["model_id"], b["seq_id"], "save_weights", b)
        return promise(rid, model_id=b["model_id"])

    @app.post("/api/v1/save_weights_for_sampler")
    async def save_weights_for_sampler(request: Request):
        b = await request.json()
        rid = eng().submit(request.state.owner, b["model_id"], b["seq_id"], "save_weights_for_sampler", b)
        return promise(rid, model_id=b["model_id"])

    @app.post("/api/v1/load_weights")
    async def load_weights(request: Request):
        b = await request.json()
        rid = eng().submit(request.state.owner, b["model_id"], b.get("seq_id", 1), "load_weights", b)
        return promise(rid, model_id=b["model_id"])

    @app.post("/api/v1/get_info")
    async def get_info(request: Request):
        b = await request.json()
        r = eng().run(request.state.owner, b["model_id"])
        return {"model_id": r.model_id, "type": "get_info", "is_lora": True, "lora_rank": r.spec.rank,
                "model_name": r.base_model, "optimizer_config": {"type": "adamw"},
                "model_data": {"model_name": r.base_model, "tokenizer_id": r.base_model}}

    @app.post("/api/v1/weights_info")
    async def weights_info(request: Request):
        b = await request.json()
        m = eng().store.meta(b["tinker_path"])
        if m.get("owner") != request.state.owner:
            raise UserError("unknown checkpoint")
        return {"base_model": m["base_model"], "is_lora": True, "lora_rank": m["lora_rank"],
                "train_attn": m["train_attn"], "train_mlp": m["train_mlp"],
                "train_unembed": m["train_unembed"], "optimizer_config": {"type": "adamw"}}

    @app.get("/api/v1/training_runs/{run_id}")
    async def training_run(run_id: str, request: Request):
        r = eng().run(request.state.owner, run_id)
        return {"training_run_id": r.model_id, "base_model": r.base_model, "model_owner": r.owner,
                "is_lora": True, "lora_rank": r.spec.rank, "user_metadata": r.user_metadata,
                "last_request_time": datetime.fromtimestamp(r.last_request, timezone.utc).isoformat()}

    @app.delete("/api/v1/training_runs/{run_id}/checkpoints/{kind}/{name}")
    async def delete_checkpoint(run_id: str, kind: str, name: str, request: Request):
        path = f"tinker://{run_id}/{kind}/{name}"
        if eng().store.meta(path).get("owner") != request.state.owner:
            raise UserError("unknown checkpoint")
        eng().store.delete(path)
        return {}

    # ---- sampling ----------------------------------------------------------------------------
    @app.post("/api/v1/create_sampling_session")
    async def create_sampling_session(request: Request):
        b = await request.json()
        return {"sampling_session_id": await eng().create_sampling_session(request.state.owner, b)}

    @app.post("/api/v1/asample")
    async def asample(request: Request):
        b = await request.json()
        rid, seq_ids = eng().submit_sample(request.state.owner, b)
        return promise(rid, sample_sequence_ids=seq_ids)

    # ---- futures -----------------------------------------------------------------------------
    @app.post("/api/v1/retrieve_future")
    async def retrieve_future(request: Request):
        b = await request.json()
        rid = b["request_id"]
        f = eng().futures
        if not f.known(rid):
            return JSONResponse({"error": "promise expired", "category": "user"}, status_code=410)
        res = await f.wait(rid, LONG_POLL_SEC)
        if res is None:
            return {"type": "try_again", "request_id": rid, "queue_state": "active"}
        if res.kind == "proto":
            return Response(res.payload, media_type="application/x-protobuf")
        return res.payload  # json result, or {"error", "category"}

    return app


def main() -> FastAPI:
    """uvicorn --factory entry point (sees every GPU; scripts/run.sh uses zroform.launch)."""
    return create_app(load_profile(os.environ.get("ZROFORM_PROFILE", "profiles/node8x-qwen38-27b-ctx32k.yaml")))
