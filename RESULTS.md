# Results: Qwen3.8-27B DeepCoder RL

The experiment behind the adapters at [ik6626/Qwen3.8-27B-deepcoder-RL](https://huggingface.co/ik6626/Qwen3.8-27B-deepcoder-RL):
its setup, the success criteria fixed before training, the results and controls, and the checks of this
repository before release. How to rerun everything: [REPRODUCE.md](REPRODUCE.md).

## Setup

| | |
|---|---|
| Model | Qwen/Qwen3.8-27B (48 Gated DeltaNet + 16 full-attention layers), text-only, BF16 |
| Adapter | LoRA rank 32, alpha 32, on attention (`q/k/v/o_proj`), Gated DeltaNet (`in_proj_qkv`, `in_proj_z`, `out_proj`), MLP and `lm_head` |
| Recipe | tinker-cookbook `code_rl`, unchanged except the grader and single turn (`scripts/code_rl.py`) |
| RL | importance-sampling policy gradient with group-relative advantages, no KL penalty; 50 problems × 4 rollouts per step; LR 1e-5; 100 steps; seeds 0 and 1 |
| Prompt | the problem as a single user message, chat template with `reasoning_effort="low"`; 12,288 tokens per answer; rollouts capped at 28,672 tokens; temperature 1 |
| Reward | 1 if the program passes all tests, else 0 (−0.1 for no code or overflow). Each program runs in a fresh, network-less, read-only container; its outputs are compared outside it (`scripts/safe_grader.py`) |
| Hardware | per seed: trainer on 1 GPU, SGLang tensor parallel 2 on 2 GPUs; 8× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, PCIe), driver 580.173.02 |
| Software | torch 2.13.0+cu130, SGLang 0.5.21, transformers 5.12.1, peft 0.21.2, flash-linear-attention 0.5.2, causal-conv1d 1.7.0; client tinker 0.33.1, tinker-cookbook 0.5.7 |
| Duration | ~23.5 min per step, ~63 h for 100 steps, both seeds in parallel; no restarts |

**Data split** (`scripts/split.py` → `data/split_manifest.json`, seed 20261022): DeepCoder's training
set has 24,287 tasks but only 18,455 distinct problems (about 5.8k are the same problem in another source),
so the split is by a hash of the normalised problem text; no copy of a held-out problem is trained on.

| Set | Source | Size | Use |
|---|---|---|---|
| Held-out A (headline) | DeepCoder train (289 primeintellect, 196 taco, 15 lcbv5), ≥ 5 tests each | 500 × 4 samples | base vs adapter |
| Held-out B | DeepCoder test, LiveCodeBench v5 part (median 42 tests) | 279 × 4 | second headline set |
| Codeforces | DeepCoder test, Codeforces part | 408 × 2 | median 1 test per problem: reported, not headline |
| Validation | DeepCoder train, disjoint from A | 100 × 1 | every 10 steps |
| Training | the rest | 23,470 tasks | RL |
| Regressions | HumanEval+ (157 × 2) and MBPP+ (375 × 2) through the same grader; IFEval, MMLU-Pro, MATH-500 (300 each, greedy) | | |

Ten HumanEval+/MBPP+ tasks are skipped because their expected values come from a reference function or a
property check rather than recorded outputs (`scripts/eval_regress.py`).

**Success criteria, fixed before training:**
1. Held-out A: adapter − base pass@1 (mean of 4 samples), paired bootstrap 95% CI over problems that
   excludes 0 in both seeds.
2. Held-out B reported the same way.
3. No regression benchmark drops more than 2 points (paired CI).
4. Healthy training (no entropy collapse, trainer–sampler logprob gap < 0.05, stable KL) and no
   reward-hacking patterns in rewarded rollouts.

**All four were met.**

## Held-out code

Final adapter vs base; pass@1, gain in points, 95% CI, problems better / worse.

| Set | Base | Seed 0 | Seed 1 | Seed 0 gain | Seed 1 gain |
|---|---|---|---|---|---|
| **Held-out A** (500) | 0.480 | 0.621 | 0.613 | **+14.1** [11.4, 16.9] (147 / 27) | **+13.3** [10.5, 16.1] (143 / 35) |
| **Held-out B**, LCB v5 (279) | 0.609 | 0.690 | 0.711 | **+8.1** [4.9, 11.3] (67 / 24) | **+10.1** [7.0, 13.4] (75 / 22) |
| Codeforces (408) | 0.212 | 0.412 | 0.395 | +20.0 [16.5, 23.4] | +18.3 [14.7, 21.8] |
| Held-out A at **step 50** | 0.480 | 0.598 | 0.598 | +11.8 [9.2, 14.5] (145 / 42) | +11.8 [9.2, 14.3] (138 / 37) |

## Regressions

| Benchmark | Base | Seed 0 | Seed 1 |
|---|---|---|---|
| HumanEval+ (157 × 2) | 0.943 | 0.955 (+1.3 [−1.0, 3.8]) | 0.943 (0.0 [−1.3, 1.3]) |
| MBPP+ (375 × 2) | 0.813 | 0.807 (−0.7 [−2.3, 0.9]) | 0.812 (−0.1 [−1.7, 1.5]) |
| IFEval (300, greedy) | 0.880 | 0.890 (+1.0 [−1.7, 3.7]) | 0.893 (+1.3 [−1.3, 4.0]) |
| MMLU-Pro (300, greedy) | 0.740 | 0.773 (+3.3 [0.3, 6.7]) | 0.777 (+3.7 [0.0, 7.3]) |
| MATH-500 (300, greedy) | 0.950 | 0.937 (−1.3 [−3.0, 0.0]) | 0.943 (−0.7 [−2.3, 1.0]) |

Nothing drops more than 2 points; MATH-500 is the only decline. The MMLU-Pro rise is probably noise at
n = 300 and isn't claimed. All general benchmarks ran with a 1,800 s per-example timeout and 0 timeouts
(the cookbook runner's default of 300 s counts an example still queued on a busy server as wrong).

## Controls

Held-out A, 500 × 4, base and seed 0 on the same server, with per-sample statistics:

| | Base | Seed 0 |
|---|---|---|
| pass@1, first run → rerun | 0.480 → 0.485 (+0.5 [−1.2, 2.2]) | 0.621 → 0.616 (−0.5 [−1.9, 0.9]) |
| Seed 0 − base, same server | | **+13.1** [10.5, 15.9] (135 / 29) |
| Samples hitting the 12,288-token limit | 18.8% | 12.8% |
| Pass rate of samples that finished | 0.597 | **0.706** |
| Median response length (tokens): all / correct | 2,421 / 1,585 | 2,039 / 1,645 |

- **Not a server artefact:** base first ran on a separate eval server (2 GPUs); rerun on a training server
  (3 GPUs) it agrees, as do both runs of seed 0. Reruns of one checkpoint differ by ±0.5–1 point
  (temperature 1).
- **Better solutions, not only finishing in time:** every sample without code is one that hit the limit.
  Changing one factor at a time from base, seed 0's completion rate alone gives about +3.6 points and its
  pass rate when finishing about +8.8, so roughly ¾ of the gain is better solutions and ¼ fewer runaway
  answers (approximate: the finished subsets differ). Correct answers did not get shorter.

## Training

| | Seed 0 | Seed 1 |
|---|---|---|
| Training reward, steps 1–10 → 91–100 | 0.488 → 0.646 | 0.478 → 0.583 |
| Validation correct (100 problems), step 0 → best → step 90 | 0.637 → 0.786 (step 50) → 0.747 | 0.655 → 0.788 (step 30) → 0.736 |
| Entropy, step 0 → 50 → 99 | 0.24 → 0.10 → 0.08 | 0.17 → 0.11 → 0.08 |
| KL (sampler vs trainer) | 0.0004 → 0.0005 | 0.0004 → 0.0005 |
| Trainer ↔ sampler logprob gap (nats/token) | 0.008–0.009 | 0.008–0.009 |

- **Health:** no entropy collapse (it falls ~3× and flattens from step 50), KL stable, gap well under 0.05.
- **Learning signal thins:** as the pass rate rises, fewer groups have both right and wrong answers (from
  ~0.35 to ~0.15–0.20 of groups), so fewer carry signal; validation plateaus after steps 30–50, while held-out
  A still gains ~2 points from step 50 to 100. Harder problems, larger groups or a curriculum would help.
- **Reward hacking:** no suspicious patterns (early exit, subprocess, file reads, directory listing,
  introspection, network, atexit/signal) in 23,773 rewarded training rollouts and ~8,900 correct evaluation
  solutions; the few matches are false positives (`sys.exit()` after printing a valid answer, a variable
  named `requests`, "socket" in a comment).

## Grader checks

| Check | Result |
|---|---|
| `scripts/grader_selftest.py`: honest programs (stdin and call-based problems, formatting edge cases), ours vs the cookbook's verdict | 23/23 identical |
| same: hostile programs (exiting early four ways incl. `os._exit(0)`, reading the answer key from disk, patching the grader's comparison, forging a result record, network egress, a fork bomb, a memory hog) | 10/10 fail |
| same: files left for the next program | none |
| `scripts/regress_selftest.py`: canonical HumanEval+ / MBPP+ solutions | 157/157, 375/375 pass; wrong and `os._exit(0)` solutions 0/20 each |
| `scripts/code_rl_selftest.py --docker --split`: single turn, no tool; the installed reward fails a wrong answer and `os._exit(0)` on a real task; no held-out or validation problem in training | all pass |

## Release checks (2026-11-01)

From a fresh copy of this repository (new venvs from the lock files; machine paths only in the gitignored
`.env`), on the hardware above:

| Check | Result |
|---|---|
| CPU tests (`pytest tests`) | all pass; 1 test skipped (needs a GPU) |
| Grader self-tests (above) | all pass |
| Adapter downloaded from the HF Hub (seed 0 final) through this repository's server: first 100 held-out A problems × 4 | **0.570**; base on the same problems 0.480 (+9.0 [4.5, 14.0], 22 better / 4 worse); the training-time evaluations of seed 0 on these problems 0.588 and 0.593 (−1.8 [−5.0, 1.3], −2.3 [−4.8, 0.3]: within sampling noise at 100 problems) |
| Stock SGLang serving the adapter converted with `convert_to_sglang.py` (as on the model card) vs this repository's server, same tokens (`scripts/check_adapter.py`; 4 problems, 7,656 answer tokens) | mean \|Δ logprob\| **0.0074** nats/token with the adapter on both; 0.049 between adapter and base; the two servers' base models differ by 0.0072 (separate 300-token check) |

So the published adapter is the one evaluated, and stock SGLang with the converted adapter computes the
same model.
