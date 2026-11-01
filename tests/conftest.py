"""CPU fixtures: a tiny random Qwen3 and a reference sampler, so the real Tinker SDK can drive the
whole server (wire, futures, ordering, LoRA train, adapter hand-off, sampling) without a GPU."""

from __future__ import annotations

import copy
import socket
import threading
import time

import pytest
import torch
import uvicorn
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict
from transformers import Qwen3Config, Qwen3ForCausalLM

from zroform.api import create_app
from zroform.config import ModelSpec, Profile
from zroform.engine import Engine
from zroform.futures import Futures
from zroform.store import LocalStore
from zroform.telemetry import MetricsLog
from zroform.trainer import Trainer
from zroform.wire import Sequence

TINY = "test/tiny-qwen3"
API_KEY = "tml-test-key"


def tiny_model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=512, tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).float()


class RefSampler:
    """Samples from a private copy of the base model with the requested adapter applied."""

    def __init__(self, base):
        self.base = base
        self.adapters: dict[str, tuple[dict, dict]] = {}
        self.loads = 0

    async def start(self):
        pass

    async def stop(self):
        pass

    async def load_adapter(self, name, tensors, config):
        self.adapters[name] = ({k: v.detach().cpu().clone() for k, v in tensors.items()}, config)
        self.loads += 1

    async def unload_adapter(self, name):
        self.adapters.pop(name, None)

    def _model(self, adapter):
        if adapter is None:
            return self.base
        tensors, cfg = self.adapters[adapter]
        m = get_peft_model(copy.deepcopy(self.base),
                           LoraConfig(**{k: v for k, v in cfg.items() if k != "peft_type"}))
        set_peft_model_state_dict(m, tensors)
        return m

    async def sample(self, tokens, n, params, adapter, prompt_logprobs):
        m = self._model(adapter).eval()
        g = torch.Generator().manual_seed(params.get("seed") or 0)
        seqs = []
        with torch.no_grad():
            for _ in range(n):
                ids, out, lps = list(tokens), [], []
                for _ in range(params.get("max_tokens") or 8):
                    logp = torch.log_softmax(m(input_ids=torch.tensor([ids])).logits[0, -1].float(), -1)
                    t = int(torch.multinomial(logp.exp(), 1, generator=g))
                    ids.append(t), out.append(t), lps.append(float(logp[t]))
                seqs.append(Sequence(out, lps, "length"))
            plp = None
            if prompt_logprobs:
                logp = torch.log_softmax(m(input_ids=torch.tensor([tokens])).logits[0].float(), -1)
                plp = [None] + [float(logp[i - 1, tokens[i]]) for i in range(1, len(tokens))]
        return seqs, plp


def make_profile(tmp_path, slots: int = 1) -> Profile:
    return Profile(name="test", slots_per_base=slots, models={TINY: ModelSpec(max_lora_rank=16)},
                   store_dir=str(tmp_path / "store"), api_keys_env="ZROFORM_TEST_KEYS")


@pytest.fixture
def server(tmp_path, monkeypatch, request):
    """Yields (base_url, engine). Parametrize indirectly with the slot count (default 1)."""
    slots = getattr(request, "param", 1)
    monkeypatch.setenv("ZROFORM_TEST_KEYS", f"{API_KEY},tml-other-tenant")
    monkeypatch.setenv("TINKER_TELEMETRY", "0")
    prof = make_profile(tmp_path, slots)
    base = tiny_model()
    sampler = RefSampler(copy.deepcopy(base))
    trainer = Trainer(TINY, slots, micro_batch_tokens=64, device="cpu", grad_checkpointing=False, model=base)
    engine = Engine(prof, trainer, sampler, LocalStore(prof.store_dir), Futures(60),
                    MetricsLog(str(tmp_path / "metrics.jsonl")))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(prof, engine), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", engine
    srv.should_exit = True
    th.join(timeout=10)
