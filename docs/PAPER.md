# Your coding-RL reward can be faked: a case study on Qwen3.8-27B

*Draft for feedback, November 2026. Code, results and adapters:
[this repository](../README.md) and [ik6626/Qwen3.8-27B-deepcoder-RL](https://huggingface.co/ik6626/Qwen3.8-27B-deepcoder-RL).*

## Abstract

The tinker-cookbook's `code_rl` recipe grades a generated program by running it inside the grading process and
treating exit code 0 as a pass, so a "solution" that calls `os._exit(0)` earns full reward; its optional
`check_solution` tool also hands the model the tests' expected outputs. Because rewards are computed on the
client side, this holds wherever the recipe runs, including on Tinker itself. We describe a grader that closes
both holes (run the program in a fresh locked-down container, judge its outputs outside it), and an
evaluation protocol that separates a real held-out gain from noise and contamination. As a worked example we
trained a rank-32 LoRA on Qwen3.8-27B with 100 RL steps on DeepCoder problems: pass@1 on 500 held-out problems
rose from 0.480 to 0.621 and 0.613 in two seeds (+14.1 and +13.3 points, paired 95% CIs above +10), and on the
LiveCodeBench v5 test set from 0.609 to 0.690 and 0.711, with no regression beyond 2 points on HumanEval+,
MBPP+, IFEval, MMLU-Pro and MATH-500. About three quarters of the gain is better solutions rather than fewer
answers running out of tokens.

## 1. What is new here, and what isn't

**Not new:** the method. The recipe is tinker-cookbook's `code_rl`, a GRPO-style policy gradient, unmodified
apart from the grader and running single-turn. RL gains of this size on competitive programming have been
reported before. Our training server (§8) is one more self-hosted implementation of the Tinker protocol;
Tinker itself is a managed service with far more models and scale, and SkyRL already provides an open-source
Tinker-compatible backend.

**What we think is useful to others:**

1. **A reward hole that affects every user of the recipe** (§2): the stock grader can be faked by exiting
   early, and the stock tool leaks answers. The fix is small and independent of where training runs.
2. **A split that accounts for duplicates** (§3): a quarter of DeepCoder's training tasks are the same problem
   in another source, so a held-out set drawn by task would leak into training. We split by problem.
3. **An evaluation protocol with controls** (§5–§6): success criteria fixed before training, paired bootstrap
   CIs over problems, a same-hardware rerun of the baseline, and a breakdown of whether the gain is better
   solutions or merely shorter answers.
4. **Two evaluation pitfalls** that each produced plausible but wrong numbers (§7), one of them in the
   cookbook's benchmark runner.
5. **Released, verified adapters**: both seeds, final and step 50, usable with transformers or (after an exact
   conversion) SGLang, with the published adapter checked against the evaluated one.

## 2. The reward hole

In coding RL the reward comes from running the model's programs: hundreds per step, written by a policy under
optimisation pressure to maximise exactly that reward. Two problems with `code_rl`'s defaults:

1. **The verdict channel is inside the program's reach.** The grader `exec`s the generated program inside the
   grading process and scores a pass by exit code 0. A program that calls `os._exit(0)` before the tests run
   scores 1.0. An existing upstream fix (PR #983) does not cover it.
2. **The tool leaks the answers.** In the recipe's two-turn mode the model may call `check_solution`, which
   runs the same tests as the reward and returns expected outputs on failure: a policy can call it once and
   hard-code the answer.

Both live on the client side of Tinker's design (the server only receives token IDs and advantages), so they
apply equally to runs on Tinker, on SkyRL or on our server.

**The fix** (`scripts/safe_grader.py`, ~400 lines) separates running from judging. Each program runs in a fresh
container (no network, read-only root, unprivileged user, no capabilities, memory/process/CPU limits, removed
afterwards) on the test inputs only, and returns its raw outputs. Comparison happens outside the container,
with the cookbook's own `run_test`, on a trusted replay of those outputs, so honest programs get exactly the
cookbook's verdict. Expected outputs never enter the container, and nothing persists between programs. For
the tool, we run single-turn with no tool at all. The function-call benchmarks (HumanEval+, MBPP+) use the
same split: the container records return values, and evalplus's own `assertion()` compares them outside.

| Check | Result |
|---|---|
| Honest programs (stdin and call-based, formatting edge cases, runtime errors, timeouts), ours vs the cookbook (`scripts/grader_selftest.py`) | 23/23 identical verdicts |
| Hostile programs: exiting early four ways (incl. `os._exit(0)`), reading the answer key from disk, patching the grader's comparison, forging a result record, network egress, a fork bomb, a memory hog | 10/10 fail |
| Files left for the next program | none |
| HumanEval+ / MBPP+ canonical solutions; wrong and `os._exit(0)` solutions (`scripts/regress_selftest.py`) | 157/157 and 375/375 pass; 0/20 each |
| Throughput | ~9 programs/s at 16 concurrent containers (container start-up bound) |

*[Before publishing: link the upstream issue or PR for the `os._exit(0)` hole.]*

## 3. A split that accounts for duplicates

DeepCoder's training set has 24,287 tasks but only 18,455 distinct problems: about 5.8k are the same problem
in another source. We split by a hash of the normalised problem text, so no copy of a held-out problem is
trained on (`scripts/split.py`, `data/split_manifest.json`):

| Set | Source | Size | Use |
|---|---|---|---|
| Held-out A (headline) | DeepCoder train, sampled by problem, ≥ 5 tests | 500 × 4 samples | base vs adapter |
| Held-out B | DeepCoder test, LiveCodeBench v5 part (median 42 tests) | 279 × 4 | second headline set |
| Codeforces | DeepCoder test, Codeforces part | 408 × 2 | median 1 test per problem: reported, not headline |
| Validation | DeepCoder train, disjoint from A | 100 × 1 | every 10 steps |
| Training | the rest | 23,470 tasks | RL |

The sources are public, so the base model may have seen some problems; that affects base and adapter alike
(the comparison is paired) but limits headroom.

## 4. Worked example: training

Qwen3.8-27B (Gated DeltaNet linear-attention layers plus full attention), text-only, BF16. LoRA rank 32 /
alpha 32 on attention, linear attention, MLP and the LM head. Importance-sampling policy gradient with
group-relative advantages, no KL penalty; 50 problems × 4 rollouts per step, LR 1e-5, 100 steps; one user
message per problem with the model's `reasoning_effort="low"` setting, 12,288 tokens per answer,
temperature 1. Per seed: one trainer GPU and two sampler GPUs, NVIDIA RTX PRO 6000 Blackwell (96 GB, PCIe);
~23.5 minutes per step, ~63 hours for both seeds in parallel.

Health: entropy fell from 0.24 to 0.08 without collapsing; KL between sampler and trainer stayed at ~0.0005 and
the per-token logprob gap at ~0.008 (bf16 noise). Validation peaked around steps 30–50 and then plateaued: as
the pass rate rose, fewer groups had both right and wrong answers (from ~0.35 to ~0.15–0.20 of groups), so
fewer carried learning signal. Held-out A still gained ~2 points between steps 50 and 100.

## 5. Evaluation protocol

Written down before training:
1. on held-out A, adapter − base pass@1 with a paired bootstrap 95% CI over problems that excludes 0 in both
   seeds;
2. held-out B reported the same way;
3. no regression benchmark dropping more than 2 points;
4. healthy training and no reward-hacking patterns in rewarded rollouts.

Evaluation uses the training environment (same prompt, grader and sampling), through the unmodified SDK.

## 6. Results

**Held-out code**, pass@1, gain in points [95% CI]:

| Set | Base | Seed 0 | Seed 1 |
|---|---|---|---|
| Held-out A (500) | 0.480 | 0.621 (+14.1 [11.4, 16.9]) | 0.613 (+13.3 [10.5, 16.1]) |
| Held-out B, LCB v5 (279) | 0.609 | 0.690 (+8.1 [4.9, 11.3]) | 0.711 (+10.1 [7.0, 13.4]) |
| Codeforces (408) | 0.212 | 0.412 (+20.0) | 0.395 (+18.3) |
| Held-out A at step 50 | 0.480 | 0.598 (+11.8) | 0.598 (+11.8) |

**Regressions**, final adapters vs base:

| Benchmark | Base | Seed 0 | Seed 1 |
|---|---|---|---|
| HumanEval+ (157 × 2) | 0.943 | 0.955 | 0.943 |
| MBPP+ (375 × 2) | 0.813 | 0.807 | 0.812 |
| IFEval (300, greedy) | 0.880 | 0.890 | 0.893 |
| MMLU-Pro (300, greedy) | 0.740 | 0.773 | 0.777 |
| MATH-500 (300, greedy) | 0.950 | 0.937 | 0.943 |

All paired CIs stay within the 2-point limit; MATH-500 is the only decline (−1.3 / −0.7). All four criteria
were met.

**Controls.**
- *Same hardware.* Base was first evaluated on a smaller eval server; rerun on a training server it scores
  0.485 vs 0.480 (+0.5 [−1.2, 2.2]), and seed 0 reruns at 0.616 vs 0.621. Seed 0 − base on one server:
  +13.1 [10.5, 15.9].
- *Better solutions or shorter answers?* Every answer without code is one that hit the token limit. Base hits
  it in 18.8% of samples, the adapter in 12.8%; the pass rate of answers that finish rises from 0.597 to 0.706.
  Changing one factor at a time from base, the completion rate alone gives about +3.6 points and the pass rate
  when finishing about +8.8: roughly three quarters of the gain is better solutions. Correct answers did not
  get shorter (median 1,585 vs 1,645 tokens).
- *Reward hacking.* No suspicious patterns (early exit, subprocess, file or directory access, introspection,
  network, signal handlers) in 23,773 rewarded training rollouts or ~8,900 correct evaluation solutions; the
  few matches were false positives.
- *The released adapter.* Served by stock SGLang after the conversion on the model card, it gives the same
  per-token logprobs as the training server to 0.0074 nats/token (noise floor), against 0.049 between adapter
  and base. Downloaded from the Hub and evaluated from a fresh clone, it scores 0.570 on the first 100 held-out
  A problems (base 0.480; the training-time runs 0.588 and 0.593 on the same problems).

## 7. Two evaluation pitfalls

1. **Timeouts scored as wrong answers.** The cookbook benchmark runner's default per-example timeout (300 s)
   counts an example still queued on a busy server as incorrect. On our smaller eval server, 172–201 of 300
   base examples timed out, and base "scored" 0.39 on IFEval and 0.32 on MATH-500; with a 1,800 s timeout and
   zero timeouts it scores 0.88 and 0.95. Check error counts before comparing.
2. **Baseline and treatment on different hardware.** Harmless here (the same-hardware control agrees), but it
   has to be checked rather than assumed.

## 8. How we ran it: Zro-form

We trained on our own GPUs with **Zro-form**, a server for the Tinker wire protocol, so the unmodified Tinker
SDK and cookbook recipes run against it (`zroform/`, ~1,800 lines):

- a **trainer**: Hugging Face transformers + PEFT LoRA on one GPU, with the Tinker loss functions implemented
  from their definitions and checked by gradient identities;
- a **sampler**: a stock SGLang server; after each `save_weights_for_sampler`, the new adapter is handed to it
  as in-memory tensors over CUDA IPC;
- an **engine**: sessions, per-model operation ordering, retry de-duplication, tenant isolation, and a
  checkpoint store with resume.

Its use is narrow: the same SDK and recipes on hardware you control, with data that never leaves it. It trains
on one GPU and has no multi-node or large mixture-of-experts support; for most users, Tinker or SkyRL is the
simpler choice, and the results in §6 should not depend on the server. Two pieces of engineering may help
others self-hosting hybrid models: PEFT's fused `in_proj_qkv` LoRA on Gated DeltaNet layers is split exactly
into SGLang's per-slice layout (the published `convert_to_sglang.py`), and the LM head's log-softmax is
computed in chunks of 1,024 positions, because full logits for one 27k-token rollout are a 13.4 GB allocation.

## 9. Limitations

- In-distribution: the held-out sets come from the same kinds of sources as training. LiveCodeBench v5 is the
  closest to an external test; Codeforces is weakly graded.
- Two seeds, one model, one dataset, one hyperparameter setting; no ablation of LoRA targets, rank, group size
  or learning rate.
- The adapter was trained and evaluated with `reasoning_effort="low"` and 12,288 answer tokens; other prompt
  settings are untested.
- The learning signal thins as the pass rate rises; harder problems, larger groups or a curriculum are the
  obvious next step.
- Single-turn only; agentic tool use with a leak-free tool is not covered.
- Temperature-1 sampling: reruns of one checkpoint differ by about ±0.5–1 point on held-out A.

## 10. Reproducibility

[REPRODUCE.md](../REPRODUCE.md) covers environment, data, grader self-tests, training, evaluation and the
expected numbers; [RESULTS.md](../RESULTS.md) has the full results and release checks. The grader and the
split work with the stock cookbook on any Tinker-compatible backend. The adapters (both seeds, final and
step 50) are on the Hugging Face Hub with transformers and SGLang instructions.

## References

- Tinker and tinker-cookbook: Thinking Machines Lab, https://github.com/thinking-machines-lab/tinker-cookbook
- SkyRL: NovaSky, https://github.com/NovaSky-AI/SkyRL
- DeepCoder-Preview-Dataset: Agentica, https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset
- LiveCodeBench: Jain et al., https://livecodebench.github.io
- EvalPlus (HumanEval+, MBPP+): Liu et al., https://github.com/evalplus/evalplus
- GRPO: Shao et al., *DeepSeekMath*, 2024
- SGLang: https://github.com/sgl-project/sglang · PEFT: https://github.com/huggingface/peft
