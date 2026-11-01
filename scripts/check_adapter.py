"""Check a served adapter end to end: this repo's server vs a stock SGLang server, on the same tokens.

For a few held-out problems (training prompts, built exactly as in training), sample one answer through our
server with the adapter, then score that same prompt + answer three ways:
  ours      per-token logprobs from our server with the adapter (the path every eval used);
  sglang    from a stock SGLang server serving the converted adapter (`--lora-paths <name>=...`);
  base      from the same SGLang server without the adapter.
The adapter is served correctly when |ours - sglang| is at the bf16 noise floor (~0.01 nats/token) and
much smaller than |sglang - base|.

  $ZROFORM_CLIENT_VENV/bin/python scripts/check_adapter.py --sampler-path tinker://imported/sampler_weights/<name> \\
      --sglang-url http://127.0.0.1:30400 --lora-name deepcoder [--problems 4]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def sglang_logprobs(url: str, tokens: list[int], start: int, lora: str | None) -> list[float]:
    body = {"input_ids": tokens, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True, "logprob_start_len": start}
    if lora:
        body["lora_path"] = lora
    req = urllib.request.Request(f"{url}/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=600))
    return [lp for lp, _tok, _txt in out["meta_info"]["input_token_logprobs"] if lp is not None]


async def main(a) -> None:
    import code_rl
    import eval_code
    import tinker
    from tinker import types as t
    from tinker_cookbook.recipes.code_rl import code_env

    os.environ.pop("ZF_SPLIT_MANIFEST", None)
    items = eval_code.load_set("heldout_a", a.manifest)[: a.problems]
    code_rl.install()
    svc = tinker.ServiceClient(base_url=a.base_url)
    sc = svc.create_sampling_client(model_path=a.sampler_path)
    rows = []
    for key, _subset, task in items:
        b = code_env.DeepcoderEnvGroupBuilder(task=task, model_name=a.model, renderer_name=a.renderer, max_turns=2,
                                              group_size=1, sandbox_backend=None)
        env = (await b.make_envs())[0]
        ob, _ = await env.initial_observation()
        prompt = ob.to_ints()
        sp = t.SamplingParams(max_tokens=a.max_tokens, temperature=1.0, seed=0)
        ans = (await sc.sample_async(t.ModelInput.from_ints(prompt), 1, sp)).sequences[0].tokens
        full = prompt + list(ans)
        ours = (await sc.compute_logprobs_async(t.ModelInput.from_ints(full)))[len(prompt):]
        # SGLang's logprob_start_len=s returns the logprobs of tokens s+1.., so ask from len(prompt) - 1
        sgl = sglang_logprobs(a.sglang_url, full, len(prompt) - 1, a.lora_name)
        base = sglang_logprobs(a.sglang_url, full, len(prompt) - 1, None)
        assert len(ours) == len(sgl) == len(base), (len(ours), len(sgl), len(base))
        n = len(ours)
        d_os = sum(abs(x - y) for x, y in zip(ours[:n], sgl[:n])) / n
        d_sb = sum(abs(x - y) for x, y in zip(sgl[:n], base[:n])) / n
        rows.append({"key": key, "answer_tokens": n, "ours_vs_sglang": round(d_os, 4), "sglang_vs_base": round(d_sb, 4)})
        print(json.dumps(rows[-1]), flush=True)
    svc.close("success").result()
    os_, sb = (sum(r[k] * r["answer_tokens"] for r in rows) / sum(r["answer_tokens"] for r in rows)
               for k in ("ours_vs_sglang", "sglang_vs_base"))
    print(json.dumps({"mean_abs_logprob_diff": {"ours_vs_sglang_adapter": round(os_, 4),
                                                "sglang_adapter_vs_base": round(sb, 4)}}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler-path", required=True)
    ap.add_argument("--sglang-url", required=True)
    ap.add_argument("--lora-name", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--renderer", default="qwen3_8_low_reasoning")
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/split_manifest.json"))
    ap.add_argument("--problems", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2048)
    asyncio.run(main(ap.parse_args()))
