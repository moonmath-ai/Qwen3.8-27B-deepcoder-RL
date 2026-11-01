"""The pinned, unmodified Tinker SDK drives our server end to end on CPU (tiny model).

Covers every verb the cookbook's RL, SFT and DPO recipes use, plus the version gate and tenant
isolation. GPU/real-model behaviour (SGLang, Qwen3-4B, cookbook recipes) is in tests/gpu.
"""

import json

import httpx
import numpy as np
import pytest
import tinker
import torch
from tinker import types as t

from tests.conftest import API_KEY, TINY


def client(base_url, key=API_KEY):
    return tinker.ServiceClient(base_url=base_url, api_key=key)


def sft_datum(tokens):
    return t.Datum(model_input=t.ModelInput.from_ints(tokens[:-1]),
                   loss_fn_inputs={"target_tokens": tokens[1:], "weights": [1.0] * (len(tokens) - 1)})


def test_rl_step_samples_trains_and_updates_sampler(server):
    url, engine = server
    tc = client(url).create_lora_training_client(base_model=TINY, rank=8)
    sc = tc.save_weights_and_get_sampling_client()
    prompt = [3, 4, 5]
    res = sc.sample(t.ModelInput.from_ints(prompt), num_samples=2,
                    sampling_params=t.SamplingParams(max_tokens=4, seed=1)).result()
    assert len(res.sequences) == 2 and all(len(s.tokens) == 4 and len(s.logprobs) == 4 for s in res.sequences)

    data = [t.Datum(model_input=t.ModelInput.from_ints(prompt + s.tokens[:-1]),
                    loss_fn_inputs={"target_tokens": [0] * (len(prompt) - 1) + s.tokens,
                                    "logprobs": [0.0] * (len(prompt) - 1) + list(s.logprobs),
                                    "advantages": [0.0] * (len(prompt) - 1) + [adv] * len(s.tokens)})
            for s, adv in zip(res.sequences, [1.0, -1.0])]
    out = tc.forward_backward(data, "importance_sampling").result()
    assert "loss:sum" in out.metrics and len(out.loss_fn_outputs) == 2

    # Trainer logprobs of the sampled tokens equal the sampler's (same weights, exact reference sampler).
    for s, o in zip(res.sequences, out.loss_fn_outputs):
        np.testing.assert_allclose(o["logprobs"].to_numpy()[len(prompt) - 1:], s.logprobs, atol=1e-4)

    tc.optim_step(t.AdamParams(learning_rate=1e-2)).result()
    loads_before = engine.sampler.loads
    tc.save_weights_and_get_sampling_client()
    assert engine.sampler.loads == loads_before + 1   # new adapter version handed to the sampler


def test_sft_loss_goes_down(server):
    url, _ = server
    tc = client(url).create_lora_training_client(base_model=TINY, rank=8)
    batch = [sft_datum([1, 2, 3, 4, 5, 6]), sft_datum([9, 8, 7, 6])]
    losses = []
    for _ in range(5):
        losses.append(tc.forward_backward(batch, "cross_entropy").result().metrics["loss:sum"])
        tc.optim_step(t.AdamParams(learning_rate=5e-2)).result()
    assert losses[-1] < losses[0]


def test_forward_only_does_not_change_weights_or_accumulate(server):
    url, _ = server
    tc = client(url).create_lora_training_client(base_model=TINY, rank=8)
    batch = [sft_datum([1, 2, 3, 4])]
    before = tc.forward(batch, "cross_entropy").result().metrics["loss:sum"]
    tc.optim_step(t.AdamParams(learning_rate=5e-2)).result()     # no gradients -> no change
    assert tc.forward(batch, "cross_entropy").result().metrics["loss:sum"] == pytest.approx(before)


def test_custom_loss_path_used_by_dpo(server):
    url, _ = server
    tc = client(url).create_lora_training_client(base_model=TINY, rank=8)

    def loss_fn(data, logprobs_list):
        loss = -sum(lp.sum() for lp in logprobs_list)
        return loss, {"custom:mean": float(loss.detach())}

    out = tc.forward_backward_custom([sft_datum([1, 2, 3, 4])], loss_fn).result()
    assert "custom:mean" in out.metrics and "loss:sum" in out.metrics


def test_save_and_load_state_round_trip(server):
    url, engine = server
    tc = client(url).create_lora_training_client(base_model=TINY, rank=8)
    batch = [sft_datum([1, 2, 3, 4, 5])]
    tc.forward_backward(batch, "cross_entropy").result()
    tc.optim_step(t.AdamParams(learning_rate=5e-2)).result()
    path = tc.save_state("ckpt").result().path
    assert path.startswith("tinker://") and path.endswith("/weights/ckpt")
    want = tc.forward(batch, "cross_entropy").result().metrics["loss:sum"]

    tc.forward_backward(batch, "cross_entropy").result()        # move away from the checkpoint
    tc.optim_step(t.AdamParams(learning_rate=5e-2)).result()
    tc.load_state_with_optimizer(path).result()
    assert tc.forward(batch, "cross_entropy").result().metrics["loss:sum"] == pytest.approx(want, rel=1e-5)


def test_rejects_other_sdk_versions(server):
    url, _ = server
    r = httpx.post(f"{url}/api/v1/client/config", json={},
                   headers={"X-API-Key": API_KEY, "X-Stainless-Package-Version": "0.32.0"})
    assert r.status_code == 400 and "tinker==0.33.1" in r.json()["error"]


@pytest.mark.parametrize("server", [3], indirect=True)
def test_tenants_cannot_see_each_other_and_do_not_interfere(server):
    url, engine = server
    a = client(url).create_lora_training_client(base_model=TINY, rank=8, seed=0)
    b = client(url, key="tml-other-tenant").create_lora_training_client(base_model=TINY, rank=8, seed=0)
    batch = [sft_datum([1, 2, 3, 4, 5])]

    # B cannot address A's run, even knowing its model_id.
    r = httpx.post(f"{url}/api/v1/optim_step",
                   json={"model_id": a.model_id, "seq_id": 50, "type": "optim_step",
                         "adam_params": {"learning_rate": 1.0}},
                   headers={"X-API-Key": "tml-other-tenant", "X-Stainless-Package-Version": "0.33.1"})
    assert r.status_code == 400 and "unknown model_id" in r.json()["error"]

    # Interleaved steps give A exactly what it gets when trained alone.
    for _ in range(3):
        a.forward_backward(batch, "cross_entropy").result()
        b.forward_backward([sft_datum([9, 9, 9, 9, 9])], "cross_entropy").result()
        a.optim_step(t.AdamParams(learning_rate=5e-2)).result()
        b.optim_step(t.AdamParams(learning_rate=5e-2)).result()
    wa = engine.trainer.adapter_tensors(engine.runs[a.model_id].slot)

    solo = client(url).create_lora_training_client(base_model=TINY, rank=8, seed=0)
    for _ in range(3):
        solo.forward_backward(batch, "cross_entropy").result()
        solo.optim_step(t.AdamParams(learning_rate=5e-2)).result()
    ws = engine.trainer.adapter_tensors(engine.runs[solo.model_id].slot)
    # Each run's optimizer holds exactly its own adapter's parameters.
    tr = engine.trainer
    for run in (a, b, solo):
        slot = engine.runs[run.model_id].slot
        owned = {id(p) for p in tr._params(slot)}
        in_opt = {id(p) for g in tr.optims[slot].param_groups for p in g["params"]}
        assert in_opt == owned
    assert wa.keys() == ws.keys()
    assert all("lora_" in k for k in wa)  # LoRA matrices only, never full base embedding weights
    for k in wa:
        torch.testing.assert_close(wa[k], ws[k])


def test_closing_a_session_frees_its_slot_immediately(server):
    url, engine = server
    svc = client(url)
    svc.create_lora_training_client(base_model=TINY, rank=8)
    assert engine.free_slots == []
    svc.close("success").result()
    assert engine.free_slots == ["s0"] and not engine.runs
    client(url).create_lora_training_client(base_model=TINY, rank=8)  # the next client gets the slot


def test_metrics_log_records_each_op_with_scalars_only(server):
    url, engine = server
    tc = client(url).create_lora_training_client(base_model=TINY, rank=8)
    sc = tc.save_weights_and_get_sampling_client()
    s = sc.sample(t.ModelInput.from_ints([3, 4, 5]), 2, t.SamplingParams(max_tokens=4, seed=1)).result().sequences
    data = [t.Datum(model_input=t.ModelInput.from_ints([3, 4, 5] + q.tokens[:-1]),
                    loss_fn_inputs={"target_tokens": [4, 5] + q.tokens, "logprobs": [0.0, 0.0] + list(q.logprobs),
                                    "advantages": [0.0, 0.0] + [a] * len(q.tokens)})
            for q, a in zip(s, [1.0, -1.0])]
    tc.forward_backward(data, "importance_sampling").result()
    tc.optim_step(t.AdamParams(learning_rate=1e-2)).result()
    lines = [json.loads(x) for x in engine.metrics.path.read_text().splitlines()]
    ops = [x["op"] for x in lines]
    assert ops[0] == "create_model" and {"sample", "forward_backward", "optim_step"} <= set(ops)
    fb = next(x for x in lines if x["op"] == "forward_backward")
    assert fb["loss_fn"] == "importance_sampling" and "logprob_gap_mean" in fb and fb["n_datums"] == 2
    for x in lines:  # every value is a number or an allow-listed identifier: no content
        assert all(isinstance(v, (int, float)) or k in ("op", "run", "loss_fn", "profile_sha256")
                   for k, v in x.items())


@pytest.mark.parametrize("loss_fn,config", [
    ("importance_sampling", None),
    ("ppo", None), ("ppo", {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}),
    ("cispo", None), ("cispo", {"clip_low_threshold": 0.5, "clip_high_threshold": 1.5}),
    ("dro", None), ("dro", {"beta": 0.5}),
])
def test_every_rl_loss_and_its_config_survive_the_sdk_round_trip(server, loss_fn, config):
    from zroform import losses as L
    url, _ = server
    tc = client(url).create_lora_training_client(base_model=TINY, rank=8)
    tokens = [3, 4, 5, 6, 7, 8, 9, 10]
    inp, tgt = tokens[:-1], tokens[1:]
    lp = torch.tensor(tc.forward([t.Datum(model_input=t.ModelInput.from_ints(inp),
                                          loss_fn_inputs={"target_tokens": tgt, "weights": [0.0] * len(tgt)})],
                                 "cross_entropy").result().loss_fn_outputs[0]["logprobs"].to_numpy())
    # Off-policy sampler logprobs: ratios exp(-delta) land both inside and outside every clip range.
    delta = torch.tensor([0.6, -0.6, 0.15, -0.15, 1.5, -1.5, 0.0])
    q, adv = lp - delta, torch.tensor([1.0, -1.0, 1.0, -1.0, 0.5, -0.5, 2.0])
    datum = t.Datum(model_input=t.ModelInput.from_ints(inp),
                    loss_fn_inputs={"target_tokens": tgt, "logprobs": q.tolist(), "advantages": adv.tolist()})
    out = tc.forward_backward([datum], loss_fn, loss_fn_config=config).result()
    got_lp = torch.tensor(out.loss_fn_outputs[0]["logprobs"].to_numpy())
    torch.testing.assert_close(got_lp, lp, atol=1e-5, rtol=0)        # forward_backward sees the same policy
    want = L.compute(loss_fn, got_lp, {"logprobs": q, "advantages": adv}, config or {})
    assert out.metrics["loss:sum"] == pytest.approx(float(want), abs=1e-4)
    if config:  # the client's config must change the answer vs. the server default
        default = L.compute(loss_fn, got_lp, {"logprobs": q, "advantages": adv}, {})
        assert float(want) != pytest.approx(float(default), abs=1e-4)


def test_cross_entropy_honours_per_token_weights_through_the_sdk(server):
    url, _ = server
    tc = client(url).create_lora_training_client(base_model=TINY, rank=8)
    inp, tgt, w = [3, 4, 5, 6], [4, 5, 6, 7], [0.0, 1.0, 0.5, 2.0]
    out = tc.forward_backward([t.Datum(model_input=t.ModelInput.from_ints(inp),
                                       loss_fn_inputs={"target_tokens": tgt, "weights": w})], "cross_entropy").result()
    lp = out.loss_fn_outputs[0]["logprobs"].to_numpy()
    assert out.metrics["loss:sum"] == pytest.approx(float(-(lp * np.array(w)).sum()), abs=1e-4)


def test_rl_losses_reject_extra_inputs_through_the_sdk(server):
    url, _ = server
    tc = client(url).create_lora_training_client(base_model=TINY, rank=8)
    bad = t.Datum(model_input=t.ModelInput.from_ints([3, 4, 5]),
                  loss_fn_inputs={"target_tokens": [4, 5, 6], "logprobs": [-1.0] * 3, "advantages": [1.0] * 3,
                                  "weights": [1.0] * 3})
    with pytest.raises(tinker.RequestFailedError, match="takes exactly"):
        tc.forward_backward([bad], "ppo").result()


def test_a_failed_forward_backward_leaves_no_gradients_for_the_next_optim_step(server, monkeypatch):
    url, engine = server
    tc = client(url).create_lora_training_client(base_model=TINY, rank=8)
    batch = [sft_datum(list(range(10, 50))), sft_datum(list(range(60, 100)))]   # 2 micro-batches (64 tokens each)
    before = {k: v.clone() for k, v in engine.trainer.adapter_tensors(engine.runs[tc.model_id].slot).items()}
    real_run, calls = engine.trainer._run, {"n": 0}

    def fail_second(idx, req):        # simulate an OOM after the first micro-batch already backpropagated
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("CUDA out of memory (simulated)")
        return real_run(idx, req)
    monkeypatch.setattr(engine.trainer, "_run", fail_second)
    with pytest.raises(tinker.RequestFailedError, match="internal error"):
        tc.forward_backward(batch, "cross_entropy").result()
    tc.optim_step(t.AdamParams(learning_rate=1.0)).result()                     # what the cookbook does next
    after = engine.trainer.adapter_tensors(engine.runs[tc.model_id].slot)
    for k in before:
        torch.testing.assert_close(after[k], before[k])
    failed = [json.loads(x) for x in engine.metrics.path.read_text().splitlines() if '"failed"' in x]
    assert failed and failed[0]["op"] == "forward_backward" and set(failed[0]) <= {"ts", "op", "run", "seq_id", "duration_s", "failed"}


@pytest.mark.parametrize("server", [2], indirect=True)
@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo", "cispo"])
def test_zero_advantage_datums_run_forward_only_with_identical_results(server, loss_fn):
    url, engine = server
    tokens = [3, 4, 5, 6, 7, 8, 9, 10]
    inp, tgt = tokens[:-1], tokens[1:]

    def batch(lp):
        q = (lp - 0.2).tolist()
        mk = lambda adv: t.Datum(model_input=t.ModelInput.from_ints(inp),           # noqa: E731
                                 loss_fn_inputs={"target_tokens": tgt, "logprobs": q, "advantages": adv})
        return [mk([1.0, -1.0, 0.5, 0.0, 2.0, -0.5, 1.0]), mk([0.0] * 7), mk([0.0] * 7)]   # 2 of 3 inert

    results = []
    for skip in (False, True):
        engine.trainer.skip_inert = skip
        tc = client(url).create_lora_training_client(base_model=TINY, rank=8, seed=0)
        lp = torch.tensor(tc.forward([t.Datum(model_input=t.ModelInput.from_ints(inp),
                                              loss_fn_inputs={"target_tokens": tgt, "weights": [0.0] * 7})],
                                     "cross_entropy").result().loss_fn_outputs[0]["logprobs"].to_numpy())
        out = tc.forward_backward(batch(lp), loss_fn).result()
        tc.optim_step(t.AdamParams(learning_rate=1e-2)).result()
        results.append((out, engine.trainer.last_inert,
                        engine.trainer.adapter_tensors(engine.runs[tc.model_id].slot)))
    (o0, n0, w0), (o1, n1, w1) = results
    assert (n0, n1) == (0, 2)                                                     # the skip actually happened
    assert o1.metrics["loss:sum"] == pytest.approx(o0.metrics["loss:sum"], abs=1e-6)
    for a, b in zip(o0.loss_fn_outputs, o1.loss_fn_outputs):                     # logprobs still returned
        np.testing.assert_allclose(a["logprobs"].to_numpy(), b["logprobs"].to_numpy(), atol=1e-6)
    for k in w0:                                                                  # identical update
        torch.testing.assert_close(w1[k], w0[k])


def test_dro_never_skips_zero_advantage_datums(server):
    url, engine = server
    tc = client(url).create_lora_training_client(base_model=TINY, rank=8)
    d = t.Datum(model_input=t.ModelInput.from_ints([3, 4, 5]),
                loss_fn_inputs={"target_tokens": [4, 5, 6], "logprobs": [-1.0] * 3, "advantages": [0.0] * 3})
    tc.forward_backward([d], "dro").result()
    assert engine.trainer.last_inert == 0                                         # beta term has a gradient


def test_a_sampling_session_from_a_checkpoint_survives_the_training_run_ending(server):
    # 2026-10-24: an eval opened a sampling client from the final checkpoint while the training session
    # was still alive, so it reused the run's live adapter; when the run was released, that adapter was
    # unloaded and every later sample failed (SGLang: "Failed to implicitly load LoRA adapter ...").
    url, engine = server
    train_svc = client(url)
    tc = train_svc.create_lora_training_client(base_model=TINY, rank=8)
    tc.forward_backward([sft_datum([1, 2, 3, 4, 5])], "cross_entropy").result()
    tc.optim_step(t.AdamParams(learning_rate=5e-2)).result()
    path = tc.save_weights_for_sampler("final").result().path
    prompt, sp = t.ModelInput.from_ints([1, 2, 3]), t.SamplingParams(max_tokens=4, temperature=1.0, seed=7)

    eval_svc = client(url)
    sc = eval_svc.create_sampling_client(model_path=path)
    before = sc.sample(prompt, 1, sp).result().sequences[0].tokens
    train_svc.close("success").result()  # releases the run and unloads its adapter versions
    after = sc.sample(prompt, 1, sp).result().sequences[0].tokens  # must reload the checkpoint, not fail
    fresh = client(url).create_sampling_client(model_path=path).sample(prompt, 1, sp).result().sequences[0].tokens
    assert after == before == fresh
