"""The cookbook's code_rl recipe with the safe grader, single turn, and the held-out split.

recipes/code_rl/train.py runs unchanged, except:
  - every grade goes through safe_grader.check_correctness (the cookbook's grader can be fooled by a
    program that exits with status 0 before the tests run);
  - single turn, no tool: the prompt is the problem only. (The cookbook's optional check_solution tool runs
    the same tests as the reward and returns expected outputs on failure.)
  - with ZF_SPLIT_MANIFEST (data/split_manifest.json from scripts/split.py): training excludes the held-out
    and validation problems, and the recipe's periodic eval (its "test" dataset) runs on the validation set.
  - checkpoint retention, which code_rl's CLI doesn't expose: ZF_ROLLING_SAVE_EVERY (a resume-only state every N
    steps; the previous one is deleted), ZF_ROLLING_TTL_SECONDS and ZF_CKPT_TTL_SECONDS ("none" = keep).
    Our server sweeps expired checkpoints every 10 s, so the recipe's 2 h rolling TTL would lose the resume
    point during a longer pause, and its 7-day periodic TTL would delete the eval checkpoints.

  $ZROFORM_CLIENT_VENV/bin/python scripts/code_rl.py base_url=... model_name=... log_path=... \
      renderer_name=qwen3_8_low_reasoning [other code_rl CLIConfig args]
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_grader  # noqa: E402
from tinker_cookbook.recipes.code_rl import code_env, deepcoder_tool  # noqa: E402
from tinker_cookbook.tool_use import build_agent_tool_env  # noqa: E402

# Cap on prompt + all turns (the cookbook builder's default is 32k). One 27B trainer GPU fits ~28k tokens per
# sequence (26-28k: 95.6 GB reserved; 30-32k: 100.9 of 101.9 GB, 2026-10-21); longer rollouts get the
# recipe's context-overflow reward instead of an OOM.
MAX_TRAJECTORY_TOKENS = int(os.environ.get("ZF_MAX_TRAJECTORY_TOKENS", "28672"))


async def _make_envs(self):
    """DeepcoderEnvGroupBuilder.make_envs, single turn with no tool: the prompt is the problem only."""
    tokenizer = code_env.tokenizer_utils.get_tokenizer(self.model_name)
    renderer = code_env.get_renderer(self.renderer_name or code_env.model_info.get_recommended_renderer_name(
        self.model_name), tokenizer)
    envs = []
    for _ in range(self.group_size):
        reward = deepcoder_tool.DeepcoderReward(task=self.task, sandbox_backend=self.sandbox_backend,
                                                timeout=self.timeout, format_coef=self.format_coef)
        envs.append(build_agent_tool_env(
            renderer=renderer, tools=[], initial_messages=[{"role": "user", "content": self.task.problem}],
            reward_fn=reward, max_trajectory_tokens=min(self.max_trajectory_tokens, MAX_TRAJECTORY_TOKENS),
            max_generation_tokens=self.max_generation_tokens,
            context_overflow_reward=self.context_overflow_reward, max_turns=1))
    return envs


def task_key(problem: str) -> str:
    """Same key as scripts/split.py (sha256 of the normalised DeepcoderTask.problem)."""
    import hashlib
    import re
    return hashlib.sha256(re.sub(r"\s+", " ", problem).strip().lower().encode()).hexdigest()[:20]


def split_loader(manifest_path: str, orig_load=None):
    """load_deepcoder_tasks honouring the split: "train" minus the excluded problems; "test" = validation."""
    orig_load = orig_load or code_env.load_deepcoder_tasks
    m = json.load(open(manifest_path))
    exclude, val = set(m["exclude_from_training"]), [v["key"] for v in m["validation"]]

    def load(split="train", seed=0):
        tasks = orig_load("train", seed=seed)
        if split == "train":
            return [t for t in tasks if task_key(t.problem) not in exclude]
        by_key = {}
        for t in tasks:
            by_key.setdefault(task_key(t.problem), t)
        return [by_key[k] for k in val]  # validation, in manifest order

    return load


def checkpoint_overrides() -> dict:
    out = {}
    if os.environ.get("ZF_ROLLING_SAVE_EVERY"):
        out["rolling_save_every"] = int(os.environ["ZF_ROLLING_SAVE_EVERY"])
    if os.environ.get("ZF_ROLLING_TTL_SECONDS"):
        out["rolling_ttl_seconds"] = int(os.environ["ZF_ROLLING_TTL_SECONDS"])
    if os.environ.get("ZF_CKPT_TTL_SECONDS"):
        v = os.environ["ZF_CKPT_TTL_SECONDS"]
        out["ttl_seconds"] = None if v.lower() == "none" else int(v)
    return out


def install() -> None:
    deepcoder_tool.sandbox_check_correctness = safe_grader.check_correctness  # the final reward
    code_env.DeepcoderEnvGroupBuilder.make_envs = _make_envs
    if os.environ.get("ZF_SPLIT_MANIFEST"):
        code_env.load_deepcoder_tasks = split_loader(os.environ["ZF_SPLIT_MANIFEST"])


if __name__ == "__main__":
    import chz
    from tinker_cookbook.recipes.code_rl import train

    install()
    overrides = checkpoint_overrides()
    if overrides:  # applied to the recipe's rl.train Config just before training starts
        real_main = train.main

        async def main_with_overrides(config):
            return await real_main(chz.replace(config, **overrides))
        train.main = main_with_overrides
    print(f"[zf] checkpoint overrides: {overrides or 'none'}", flush=True)
    print(f"[zf] code_rl with safe_grader, single turn, grader image {safe_grader.IMAGE}, "
          f"max trajectory {MAX_TRAJECTORY_TOKENS} tokens, split {os.environ.get('ZF_SPLIT_MANIFEST') or 'none'}", flush=True)
    asyncio.run(train.cli_main(chz.entrypoint(train.CLIConfig)))
