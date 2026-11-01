"""Self-test for scripts/code_rl.py: the recipe uses the safe grader, single turn, and the split.

  $ZROFORM_CLIENT_VENV/bin/python scripts/code_rl_selftest.py [--docker] [--split data/split_manifest.json]

Always: the environment offers no tool, one turn, and a prompt that is the problem only.
With --docker: on a real DeepCoder (primeintellect) task, the installed reward grader fails a wrong answer
and an os._exit(0) "solution".
With --split: no held-out or validation problem reaches training, and the recipe's eval set is the
validation set.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_rl  # noqa: E402
from tinker_cookbook.recipes.code_rl.deepcoder_tool import DeepcoderTask  # noqa: E402

TASK = DeepcoderTask(problem="Print twice the input.\n", tests=[{"input": "3\n", "output": "6\n", "testtype": "stdin"}])


async def env_check() -> None:
    captured = {}
    code_rl.build_agent_tool_env = lambda **kw: captured.update(kw) or object()

    class Builder:  # the attributes _make_envs reads from DeepcoderEnvGroupBuilder
        task, model_name, renderer_name, max_turns, group_size = TASK, "Qwen/Qwen3.8-27B", "qwen3_8_low_reasoning", 2, 1
        sandbox_backend, timeout, format_coef = None, 6, 0.1
        max_trajectory_tokens, max_generation_tokens, context_overflow_reward = 32768, None, -0.1

    await code_rl._make_envs(Builder())
    assert captured["tools"] == [] and captured["max_turns"] == 1
    assert captured["initial_messages"] == [{"role": "user", "content": TASK.problem}]
    assert captured["max_trajectory_tokens"] == min(32768, code_rl.MAX_TRAJECTORY_TOKENS) == 28672
    print("ok  env: no tool offered, one turn, prompt is the problem only, rollouts capped at 28,672 tokens")


async def docker_check() -> None:
    import importlib
    importlib.reload(code_rl)  # undo the fake above
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from tinker_cookbook.recipes.code_rl import deepcoder_tool
    code_rl.install()
    assert deepcoder_tool.sandbox_check_correctness is code_rl.safe_grader.check_correctness
    snap = snapshot_download("agentica-org/DeepCoder-Preview-Dataset", repo_type="dataset")
    row = pq.read_table(sorted(glob.glob(f"{snap}/primeintellect/train-*.parquet"))[0]).slice(0, 1).to_pylist()[0]
    tests = json.loads(row["tests"])
    tests = [{"input": t["input"], "output": t["output"], "testtype": "stdin"} for t in tests] if isinstance(tests, list) and "input" in tests[0] else tests
    for name, code in [("wrong answer", "print('definitely wrong')"), ("os._exit(0)", "import os\nos._exit(0)\n")]:
        passed, _ = await deepcoder_tool.sandbox_check_correctness(tests, code, timeout=6)
        assert passed is False, name
        print(f"ok  docker: real task ({len(tests)} tests), {name} -> fail")


def split_check(manifest_path: str) -> None:
    """The split-aware loader on the real dataset: no held-out or validation problem reaches training."""
    import importlib
    importlib.reload(code_rl)
    import split
    m = json.load(open(manifest_path))
    load = code_rl.split_loader(manifest_path)
    train, val = load("train", seed=0), load("test", seed=0)
    train_keys = {code_rl.task_key(t.problem) for t in train}
    held = {h["key"] for h in m["heldout_a"]} | {v["key"] for v in m["validation"]}
    assert all(code_rl.task_key(t.problem) == split.task_key(t.problem) for t in train[:200])
    assert not train_keys & set(m["exclude_from_training"]) and not train_keys & held
    assert len(train) == m["counts"]["training_tasks_after_exclusion"], (len(train), m["counts"])
    assert [code_rl.task_key(t.problem) for t in val] == [v["key"] for v in m["validation"]]
    print(f"ok  split: {len(train)} training tasks, none of the {len(held)} held-out/validation problems; "
          f"eval set = the {len(val)} validation problems")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docker", action="store_true")
    ap.add_argument("--split", help="split manifest to check the training/validation filter against")
    a = ap.parse_args()
    asyncio.run(env_check())
    if a.docker:
        asyncio.run(docker_check())
    if a.split:
        split_check(a.split)
    print("all checks passed")
