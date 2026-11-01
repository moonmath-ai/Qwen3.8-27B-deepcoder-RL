"""Per-run RL training-health report: client metrics (cookbook) + server metrics (zroform telemetry).

Answers "is training healthy?" step by step, and flags known failure patterns. Thresholds are
starting heuristics (2026-10-18), to be tuned on real runs.

  python3 scripts/health_report.py --client <log_path>/metrics.jsonl \
      --server $ZROFORM_DATA/metrics/server_<ts>.jsonl [--out report.md]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

# Heuristic thresholds (starting heuristics, tuned on real runs).
MIN_SIGNAL = 0.2          # by_group/frac_mixed: share of groups whose samples disagree (others give zero advantage)
STALL_STEPS = 3           # consecutive low-signal steps before flagging
ENTROPY_COLLAPSE = 0.3    # entropy below 30% of its early mean
LENGTH_DRIFT = 2.0        # mean response length x2 or /2 vs early steps
MAX_KL_SAMPLE_TRAIN = 0.01
MAX_LOGPROB_GAP = 0.05


def _find(row: dict, suffix: str):
    """Cookbook keys carry prefixes (env/all/...); match on the suffix."""
    for k, v in row.items():
        if k == suffix or k.endswith("/" + suffix):
            return v
    return None


def load_client(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    out = []
    for i, r in enumerate(rows):
        out.append({"step": r.get("step", i), "reward": _find(r, "reward/total") or _find(r, "total"),
                    "frac_mixed": _find(r, "by_group/frac_mixed"),
                    "frac_all_good": _find(r, "by_group/frac_all_good"),
                    "frac_all_bad": _find(r, "by_group/frac_all_bad"),
                    "entropy": _find(r, "optim/entropy"),
                    "kl_sample_train": _find(r, "optim/kl_sample_train_v1"),
                    "resp_len": _find(r, "ac_tokens_per_turn"), "time": _find(r, "time/total")})
    return out


def load_server(path: Path | None, run: str | None) -> list[dict]:
    """One entry per optim_step: datum-weighted logprob gap and loss over that step's forward_backward calls."""
    if not path:
        return []
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    run = run or Counter(x.get("run") for x in lines if x.get("op") == "optim_step").most_common(1)[0][0]
    steps, gaps, n = [], [], 0
    for x in lines:
        if x.get("run") != run:
            continue
        if x["op"] == "forward_backward" and "logprob_gap_mean" in x:
            gaps.append((x["logprob_gap_mean"], x.get("n_datums", 1)))
        elif x["op"] == "optim_step":
            gap = sum(g * w for g, w in gaps) / sum(w for _, w in gaps) if gaps else None
            steps.append({"step": n, "logprob_gap": gap, "grad_norm": x.get("grad_norm"),
                          "gpu_reserved_gb": x.get("gpu_reserved_gb")})
            gaps, n = [], n + 1
    return steps


def flags(client: list[dict], server: list[dict]) -> list[str]:
    out = []
    sig = [r["frac_mixed"] for r in client]
    run_len = 0
    for i, s in enumerate(sig):
        run_len = run_len + 1 if s is not None and s < MIN_SIGNAL else 0
        if run_len == STALL_STEPS:
            out.append(f"STALLED from step {i - STALL_STEPS + 2}: frac_mixed < {MIN_SIGNAL} for {STALL_STEPS} steps "
                       "(most groups all-right or all-wrong -> little learning signal; task too easy/hard?)")
    for key, label in (("entropy", "entropy"), ("resp_len", "response length")):
        vals = [r[key] for r in client if r[key] is not None]
        if len(vals) >= 8:
            early = st.mean(vals[:5])
            if key == "entropy" and vals[-1] < ENTROPY_COLLAPSE * early:
                out.append(f"ENTROPY COLLAPSE: {early:.3f} -> {vals[-1]:.3f} (policy turning deterministic)")
            if key == "resp_len" and early > 0 and not (1 / LENGTH_DRIFT <= vals[-1] / early <= LENGTH_DRIFT):
                out.append(f"LENGTH DRIFT: {early:.0f} -> {vals[-1]:.0f} tokens (possible reward hacking)")
    kl = [r["kl_sample_train"] for r in client if r["kl_sample_train"] is not None]
    if kl and max(kl) > MAX_KL_SAMPLE_TRAIN:
        out.append(f"SAMPLER/TRAINER MISMATCH: kl_sample_train max {max(kl):.4f} > {MAX_KL_SAMPLE_TRAIN}")
    gap = [s["logprob_gap"] for s in server if s["logprob_gap"] is not None]
    if gap and st.mean(gap) > MAX_LOGPROB_GAP:
        out.append(f"SAMPLER/TRAINER MISMATCH: mean logprob gap {st.mean(gap):.4f} > {MAX_LOGPROB_GAP}")
    rew = [r["reward"] for r in client if r["reward"] is not None]
    if len(rew) >= 15 and st.mean(rew[-10:]) <= st.mean(rew[:5]):
        out.append(f"NO IMPROVEMENT: last-10 reward {st.mean(rew[-10:]):.3f} <= first-5 {st.mean(rew[:5]):.3f}")
    return out


def report(client: list[dict], server: list[dict]) -> str:
    by_step = {s["step"]: s for s in server}
    cols = ["step", "reward", "frac_mixed", "entropy", "kl_sample_train", "resp_len", "logprob_gap", "grad_norm",
            "time"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    fmt = lambda v: "–" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
    for r in client:
        s = by_step.get(r["step"], {})
        lines.append("| " + " | ".join(fmt({**r, **s}.get(c)) for c in cols) + " |")
    fl = flags(client, server)
    lines += ["", "**Flags:** " + ("none" if not fl else ""), *[f"- {f}" for f in fl]]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True, type=Path)
    ap.add_argument("--server", type=Path)
    ap.add_argument("--run", help="server run id (default: the one with most optim steps)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    md = report(load_client(a.client), load_server(a.server, a.run))
    print(md)
    if a.out:
        a.out.write_text(md + "\n")
