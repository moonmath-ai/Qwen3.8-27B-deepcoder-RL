"""Chunked target logprobs equal the direct log-softmax, in value and gradient; batches respect padding."""

import numpy as np
import torch

from zroform import trainer as tr
from zroform.wire import Datum


def test_chunked_logprobs_match_full_log_softmax_in_value_and_gradient(monkeypatch):
    monkeypatch.setattr(tr, "LOGPROB_CHUNK", 7)                 # force several uneven chunks
    torch.manual_seed(0)
    logits = torch.randn(30, 50, dtype=torch.bfloat16)
    tgt = torch.randint(0, 50, (30,))
    a = logits.clone().requires_grad_(True)
    b = logits.clone().requires_grad_(True)
    want = torch.log_softmax(a.float(), -1).gather(-1, tgt[:, None]).squeeze(-1)
    got = tr.token_logprobs(b, tgt)
    torch.testing.assert_close(got, want)
    w = torch.randn(30)
    (want * w).sum().backward()
    (got * w).sum().backward()
    torch.testing.assert_close(b.grad, a.grad)


def test_micro_batches_bound_padded_tokens_and_cover_every_datum_once():
    lens = [10, 900, 50, 3000, 40, 5000, 20]                     # 5000 exceeds the budget on its own
    t = tr.Trainer.__new__(tr.Trainer)
    t.micro_batch_tokens = 1000
    data = [Datum(tokens=np.zeros(n, np.int64), inputs={}) for n in lens]
    batches = list(t._micro_batches(data))
    assert sorted(i for b in batches for i in b) == list(range(len(lens)))
    for b in batches:
        assert len(b) == 1 or len(b) * max(lens[i] for i in b) <= 1000


def test_chunked_lm_head_matches_full_logits_in_value_and_lora_gradient(monkeypatch):
    # The trainer never materialises [T, V] logits; per-chunk lm_head + log-softmax must give the same
    # logprobs and the same gradients for the LoRA weights (incl. lm_head's own LoRA) as the full path.
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    monkeypatch.setattr(tr, "LOGPROB_CHUNK", 5)
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=False)
    model = get_peft_model(Qwen3ForCausalLM(cfg), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "down_proj", "lm_head"]))
    for n, p in model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.1)  # non-zero B so the adapter (and lm_head's LoRA) matters
    ids, tgt = torch.randint(0, 97, (1, 23)), torch.randint(0, 97, (23,))

    def grads(lp):
        model.zero_grad()
        (lp * torch.linspace(-1, 1, 23)).sum().backward()
        return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}

    want = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1).gather(-1, tgt[:, None]).squeeze(-1)
    g_want = grads(want)
    base = model.get_base_model()
    got = tr.head_logprobs(base.lm_head, base.model(input_ids=ids).last_hidden_state[0], tgt)
    g_got = grads(got)
    torch.testing.assert_close(got, want)
    assert g_got.keys() == g_want.keys() and any("lm_head" in k for k in g_got)
    for k in g_want:
        torch.testing.assert_close(g_got[k], g_want[k], msg=k)
