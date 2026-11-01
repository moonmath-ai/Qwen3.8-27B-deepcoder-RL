"""Held-out code eval: base vs adapter, same protocol as training.

Runs each problem of a held-out set N times through the recipe's own environment, via scripts/code_rl.py:
the same single-turn prompt, safe grader, rollout cap and sampling policy as RL
training (TinkerTokenCompleter, max_tokens, temperature 1). A sample is correct iff its reward is 1
(rewards: 1 correct, 0 wrong, -0.1 no code or overflow).

  run:     $ZROFORM_CLIENT_VENV/bin/python scripts/eval_code.py run --set heldout_a --n 4 \\
               [--sampler-path tinker://<run>/sampler_weights/<name>] --out <dir>
  compare: $ZROFORM_CLIENT_VENV/bin/python scripts/eval_code.py compare <base_dir> <adapter_dir>

Sets: heldout_a (500, the headline), heldout_b (DeepCoder test, lcbv5; median 42 tests), codeforces
(DeepCoder test, Codeforces; median 1 test, so weakly graded), validation (100).
Outputs: results.jsonl (one line per problem: key, subset, n, correct) and correct_code.jsonl (the final
code of correct samples, for the reward-hacking scan) and samples.jsonl (per sample: ac_len, completed, format).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/split_manifest.json")


def load_set(name: str, manifest_path: str):
    """[(key, subset, DeepcoderTask)] for a held-out set, built exactly as the recipe builds tasks."""
    import split
    from tinker_cookbook.recipes.code_rl import code_env
    m = json.load(open(manifest_path))
    if name in ("heldout_a", "validation"):
        wanted = [(x["key"], x["subset"]) for x in m[name]]
        tasks = code_env.load_deepcoder_tasks("train", seed=0)
    else:
        sub = "lcbv5" if name == "heldout_b" else "codeforces"
        wanted = [(r["key"], r["subset"]) for r in split.load("test") if r["subset"] == sub]
        tasks = code_env.load_deepcoder_tasks("test", seed=0)
    by_key = {}
    for t in tasks:
        by_key.setdefault(split.task_key(t.problem), t)
    return [(k, s, by_key[k]) for k, s in wanted]


async def run(a) -> None:
    import code_rl
    import tinker
    from tinker_cookbook import model_info
    from tinker_cookbook.completers import TinkerTokenCompleter
    from tinker_cookbook.recipes.code_rl import code_env
    from tinker_cookbook.rl.rollouts import do_group_rollout

    os.environ.pop("ZF_SPLIT_MANIFEST", None)  # that filter removes exactly these problems from the loader
    items = load_set(a.set, a.manifest)        # before install(), from the unfiltered loader
    code_rl.install()  # safe grader + leak-free tool, as in training
    os.makedirs(a.out, exist_ok=True)
    if a.limit:
        items = items[: a.limit]
    svc = tinker.ServiceClient(base_url=a.base_url)
    sc = (svc.create_sampling_client(model_path=a.sampler_path) if a.sampler_path
          else svc.create_sampling_client(base_model=a.model))
    policy = TinkerTokenCompleter(sc, max_tokens=a.max_tokens, temperature=1.0)
    renderer = a.renderer or model_info.get_recommended_renderer_name(a.model)
    sem = asyncio.Semaphore(a.concurrency)
    done = {json.loads(l)["key"] for l in open(f"{a.out}/results.jsonl")} if os.path.exists(f"{a.out}/results.jsonl") else set()
    res_f, code_f = open(f"{a.out}/results.jsonl", "a"), open(f"{a.out}/correct_code.jsonl", "a")
    samp_f = open(f"{a.out}/samples.jsonl", "a")
    t0, n_done = time.time(), len(done)

    async def one(key, subset, task):
        nonlocal n_done
        builder = code_env.DeepcoderEnvGroupBuilder(task=task, model_name=a.model, renderer_name=renderer, max_turns=2,
                                                    group_size=a.n, sandbox_backend=None)
        async with sem:
            group = await do_group_rollout(builder, policy)
        rewards = group.get_total_rewards()
        correct = [r > 0.5 for r in rewards]
        res_f.write(json.dumps({"key": key, "subset": subset, "n": len(rewards), "correct": sum(correct),
                                "rewards": rewards}) + "\n"); res_f.flush()
        for i, (traj, ok) in enumerate(zip(group.trajectories_G, correct)):
            last = traj.transitions[-1] if traj.transitions else None  # per-sample length / truncation / format
            m = (last.metrics or {}) if last else {}
            samp_f.write(json.dumps({"key": key, "sample": i, "correct": ok, "ac_len": len(last.ac.tokens) if last else 0,
                                     "completed": m.get("stop/completed", 0.0), "format": m.get("format", 0.0)}) + "\n")
            if ok:
                text = (traj.transitions[-1].logs or {}).get("assistant_content", "") if traj.transitions else ""
                code_f.write(json.dumps({"key": key, "sample": i, "final": text[-20000:]}) + "\n")
        code_f.flush(); samp_f.flush()
        n_done += 1
        if n_done % 25 == 0:
            print(f"[eval] {n_done}/{len(items)} problems, {time.time() - t0:.0f} s", flush=True)

    await asyncio.gather(*[one(k, s, t) for k, s, t in items if k not in done])
    svc.close("success").result()
    summarize(a.out)


def summarize(d: str) -> dict:
    rows = [json.loads(l) for l in open(f"{d}/results.jsonl")]
    p = [r["correct"] / r["n"] for r in rows]
    out = {"problems": len(rows), "samples": sum(r["n"] for r in rows), "pass@1": round(sum(p) / len(p), 4)}
    json.dump(out, open(f"{d}/summary.json", "w"), indent=1)
    print(json.dumps(out))
    return out


def compare(base: str, adapter: str, n_boot: int = 10000, seed: int = 0, key: str = "key") -> dict:
    """Paired by problem: mean of (adapter pass@1 - base pass@1), bootstrap 95% CI over problems."""
    def load(d):
        return {r[key]: r["correct"] / r["n"] for r in map(json.loads, open(f"{d}/results.jsonl"))}
    b, a = load(base), load(adapter)
    keys = sorted(b.keys() & a.keys())
    diffs = [a[k] - b[k] for k in keys]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
    out = {"problems": len(keys), "base": round(sum(b[k] for k in keys) / len(keys), 4),
           "adapter": round(sum(a[k] for k in keys) / len(keys), 4), "diff": round(sum(diffs) / len(diffs), 4),
           "ci95": [round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot)], 4)],
           "improved": sum(x > 0 for x in diffs), "worse": sum(x < 0 for x in diffs)}
    out["ci_excludes_0"] = out["ci95"][0] > 0 or out["ci95"][1] < 0
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", required=True, choices=["heldout_a", "heldout_b", "codeforces", "validation"])
    r.add_argument("--n", type=int, default=4)
    r.add_argument("--out", required=True)
    r.add_argument("--sampler-path", help="tinker:// sampler weights of the adapter; omit for the base model")
    r.add_argument("--base-url", default="http://127.0.0.1:8100")
    r.add_argument("--model", default="Qwen/Qwen3.8-27B")
    r.add_argument("--renderer", default="qwen3_8_low_reasoning")
    r.add_argument("--max-tokens", type=int, default=12288)
    r.add_argument("--concurrency", type=int, default=64, help="problems in flight")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--manifest", default=MANIFEST)
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("adapter")
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run(a))
    else:
        compare(a.base, a.adapter)
