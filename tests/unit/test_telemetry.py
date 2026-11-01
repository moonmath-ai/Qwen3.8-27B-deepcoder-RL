"""The metrics log keeps only allow-listed scalars and fixed-format identifiers."""

import json

from zroform.telemetry import MetricsLog, clean


def test_drops_unknown_fields_free_text_and_non_finite_numbers():
    got = clean({"loss_sum": 1.5, "n_tokens": 12, "grad_norm": float("nan"),
                 "prompt": "secret text", "loss_fn": "importance_sampling", "run": "zf-0123456789ab:train:0",
                 "error": "boom", "seq_id": "7", "logprob_gap_mean": True, "profile_sha256": "not-a-hash"})
    assert got == {"loss_sum": 1.5, "n_tokens": 12, "loss_fn": "importance_sampling",
                   "run": "zf-0123456789ab:train:0"}


def test_writes_one_json_line_per_known_op_and_ignores_unknown_ops(tmp_path):
    log = MetricsLog(str(tmp_path / "m.jsonl"))
    log.record("optim_step", grad_norm=2.0, tokens=[1, 2, 3])
    log.record("dump_everything", loss_sum=1.0)
    lines = [json.loads(x) for x in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert len(lines) == 1 and lines[0]["op"] == "optim_step" and "tokens" not in lines[0]
