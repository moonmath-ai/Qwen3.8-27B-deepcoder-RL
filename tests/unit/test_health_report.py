"""Health report flags known RL failure patterns and stays quiet on a healthy run."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("hr", Path(__file__).parents[2] / "scripts/health_report.py")
hr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hr)


def _client(tmp_path, rows):
    p = tmp_path / "metrics.jsonl"
    p.write_text("\n".join(json.dumps({"step": i, **r}) for i, r in enumerate(rows)))
    return hr.load_client(p)


def _row(reward, mixed=0.6, ent=0.8, length=200, kl=0.001):
    return {"env/all/reward/total": reward, "env/all/by_group/frac_mixed": mixed, "optim/entropy": ent,
            "env/all/ac_tokens_per_turn": length, "optim/kl_sample_train_v1": kl}


def test_healthy_run_has_no_flags(tmp_path):
    rows = [_row(0.3 + 0.02 * i) for i in range(20)]
    assert hr.flags(_client(tmp_path, rows), []) == []


def test_flags_stall_entropy_collapse_length_drift_mismatch_and_no_improvement(tmp_path):
    rows = [_row(0.5, mixed=0.1 if i > 5 else 0.6, ent=0.8 if i < 10 else 0.1,
                 length=200 if i < 10 else 900, kl=0.05 if i == 12 else 0.001) for i in range(20)]
    text = " ".join(hr.flags(_client(tmp_path, rows), []))
    for word in ("STALLED", "ENTROPY COLLAPSE", "LENGTH DRIFT", "MISMATCH", "NO IMPROVEMENT"):
        assert word in text


def test_server_steps_weight_logprob_gap_by_datums(tmp_path):
    p = tmp_path / "server.jsonl"
    run = "zf-0123456789ab:train:0"
    p.write_text("\n".join(json.dumps(x) for x in [
        {"op": "forward_backward", "run": run, "logprob_gap_mean": 0.01, "n_datums": 3},
        {"op": "forward_backward", "run": run, "logprob_gap_mean": 0.05, "n_datums": 1},
        {"op": "optim_step", "run": run, "grad_norm": 2.0}]))
    (step,) = hr.load_server(p, None)
    assert abs(step["logprob_gap"] - 0.02) < 1e-9 and step["grad_norm"] == 2.0
