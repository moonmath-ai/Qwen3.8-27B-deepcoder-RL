# Reproducing the Qwen3.8-27B DeepCoder RL result

End to end: environment, data, the safe grader, training (100 steps × 2 seeds), held-out and regression
evals, and the paired comparisons. Setup, success criteria, results and caveats: [RESULTS.md](RESULTS.md). The trained adapters are on the
HF Hub, public: [ik6626/Qwen3.8-27B-deepcoder-RL](https://huggingface.co/ik6626/Qwen3.8-27B-deepcoder-RL); §7 evaluates them without retraining.

## 0. What you need

| | Used for the result | Notes |
|---|---|---|
| GPUs | 8× RTX PRO 6000 Blackwell (96 GB, sm_120, PCIe) | One seed needs 3 GPUs (trainer on 1, SGLang TP=2 on 2). The launcher's layout: seed 0 on GPUs 0–2, seed 1 on 4–6, an eval server on 3 + 7. One seed alone works on 3 GPUs. |
| Disk | ~55 GB model, ~38 GB checkpoints per seed, a few GB of data and rollouts | |
| Docker | the user must be in the `docker` group | the grader runs every program in a fresh container |
| Python | 3.12, [`uv`](https://docs.astral.sh/uv/) | two venvs (server and client); they **must be separate**: the cookbook's transformers cap conflicts with SGLang's pin |
| Time | training ~63 h (both seeds in parallel, ~23.5 min per step); base evals ~1 day on the eval server; adapter evals ~4–5 h per seed | |

Versions of the result: torch 2.13.0+cu130, SGLang 0.5.21, transformers 5.12.1, peft 0.21.2,
flash-linear-attention 0.5.2, causal-conv1d 1.7.0 (server); tinker 0.33.1, tinker-cookbook 0.5.7
(client); NVIDIA driver 580.173.02.

## 1. Environment

```bash
git clone https://github.com/moonmath-ai/Qwen3.8-27B-deepcoder-RL && cd Qwen3.8-27B-deepcoder-RL

# Server venv (.venv). The kernels extra builds causal-conv1d from source against the installed torch;
# set the arch for your GPUs (12.0 = Blackwell RTX PRO 6000). Without it the trainer falls back to
# PyTorch for that op (slower, same numerics within noise; zroform/kernels.py).
TORCH_CUDA_ARCH_LIST=12.0 CAUSAL_CONV1D_FORCE_BUILD=TRUE uv sync --extra kernels

# Client venv (SDK + cookbook recipes, evals): .venv-client, the default location
uv venv .venv-client -p 3.12
uv pip install -p .venv-client -r requirements-client.txt   # exact pins of the run (tinker 0.33.1, cookbook 0.5.7)

# Machine paths (optional): copy and edit; .env is gitignored
cp .env.example .env
source scripts/env.sh     # in every shell where you run the Python tools by hand; the shell scripts do it themselves
```

`scripts/env.sh` reads `.env` and gives every path a default inside the repo: run outputs and checkpoints
under `runs/`, the client venv at `.venv-client/`. The only setting most machines need is `HF_HUB_CACHE`,
if the model is already cached somewhere other than `~/.cache/huggingface`. The scripts unset `HF_TOKEN`
for child processes (model and data are ungated).

CPU tests (no GPU needed): `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests -q` (78 pass, 1 skip).

## 2. Model and data (downloaded into the HF cache on first use)

| What | HF repo | Used by |
|---|---|---|
| Model | `Qwen/Qwen3.8-27B` (text-only use) | server profiles |
| Training + held-out A, validation | `agentica-org/DeepCoder-Preview-Dataset` configs `primeintellect`, `taco`, `lcbv5` (revision `177913a7bd43791646ef6a43645caa3c871ab3db` was used) | `scripts/code_rl.py`, `eval_code.py` |
| Held-out B, Codeforces | the same dataset's `lcbv5` and `codeforces` test splits | `eval_code.py` |
| Coding regressions | `evalplus/humanevalplus`, `evalplus/mbppplus` | `eval_regress.py` |
| General regressions | `google/IFEval`, `TIGER-Lab/MMLU-Pro`, `HuggingFaceH4/MATH-500` | `scripts/eval_bench.py` |

The held-out / validation / training split is fixed in `data/split_manifest.json` (by normalised
problem text, near-duplicates included; `scripts/split.py`, seed 20261022 regenerates it).

## 3. The safe grader (do this before training)

The cookbook's `code_rl` grader scores a program by its exit code inside the grader process, so
`os._exit(0)` earns full reward, and its `check_solution` tool reveals expected outputs. We replace both
(`scripts/safe_grader.py`): each program runs in a fresh locked-down
container and its outputs are compared **outside** it.

```bash
docker build -t zf-code-runner:1 -f scripts/runner.Dockerfile .
source scripts/env.sh
$ZROFORM_CLIENT_VENV/bin/python scripts/grader_selftest.py            # 23/23 honest agree, 10/10 exploits fail, no persistence
$ZROFORM_CLIENT_VENV/bin/python scripts/regress_selftest.py           # canonical 157/157 and 375/375 pass; wrong and os._exit fail
$ZROFORM_CLIENT_VENV/bin/python scripts/code_rl_selftest.py --docker --split data/split_manifest.json  # single turn, safe reward, split
```

If you joined the `docker` group after logging in, run these through `sg docker -c "..."` (the shell
scripts detect this themselves).

## 4. Train: 100 steps × 2 seeds

```bash
nvidia-smi                          # the launcher refuses to start if its GPUs are busy
scripts/main_run.sh start        # servers, both seeds, base evals, supervisor; all in tmux
scripts/main_run.sh status       # steps, rewards, validation, checkpoints, GPUs
scripts/main_run.sh stop         # clean stop (loses at most the step in progress)
scripts/main_run.sh resume       # continues from the last per-step rolling checkpoint
```

- **Settings** (in `main_run.sh`): LoRA rank 32, LR 1e-5, importance-sampling loss, 50 groups × 4
  rollouts per step, 100 steps, renderer `qwen3_8_low_reasoning`, 12,288 tokens per answer, single turn,
  no tool, validation every 10 steps, permanent checkpoints every 10 steps.
- **GPU layout** is set at the top of `main_run.sh` (`GPUS_0/1/E`) and in the profiles
  (`profiles/node8x-qwen38-27b-ctx32k*.yaml`: `trainer_device`, `sampler_gpus`). Change both together.
- The supervisor restarts crashed or hung runs from their rolling checkpoint and, when a seed finishes,
  starts that seed's adapter evals on its server. A disk guard stops everything below 20 GB free.
- Outputs: `$ZROFORM_OUT_ROOT/main/` (`seed0/`, `seed1/`, `evals/`); checkpoints in
  `$ZROFORM_DATA/store/<run>/{weights,sampler_weights}/<step>`.

## 5. Evaluate

The launcher queues these automatically; to run them by hand on a server's port:

```bash
R=$ZROFORM_OUT_ROOT/main/evals
SP=tinker://<run>/sampler_weights/final          # from seed<s>/checkpoints.jsonl ("name": "final")
scripts/eval_queue.sh 8300 $R base                       # held-out A ×4, held-out B ×4, Codeforces ×2
scripts/eval_queue.sh 8100 $R adapter_seed0 $SP
scripts/eval_queue_regress.sh 8300 $R base_regress       # HumanEval+, MBPP+, IFEval, MMLU-Pro, MATH-500
scripts/eval_queue_regress.sh 8100 $R adapter_seed0_regress $SP
```

Paired comparisons (bootstrap 95% CI over problems):

```bash
$ZROFORM_CLIENT_VENV/bin/python scripts/eval_code.py compare $R/base_heldout_a $R/adapter_seed0_heldout_a
$ZROFORM_CLIENT_VENV/bin/python scripts/eval_regress.py compare $R/base_regress_humanevalplus $R/adapter_seed0_regress_humanevalplus
```

**Before comparing, check validity** (each of these cost us a rerun):
- every set reached its full count (500 / 279 / 408; 157 / 375; 300): `.done` markers are written even
  when a set failed;
- `errors` (timeouts) in `*_bench/bench.json` are ~0: the cookbook's default 300 s per example scores
  queueing as a wrong answer (`eval_bench.py --timeout`, default 1800);
- base and adapter ran under the same conditions (the result was also checked on one server: RESULTS.md).

Training health: `$ZROFORM_CLIENT_VENV/bin/python scripts/health_report.py --client $R/../seed0/metrics.jsonl`.

## 6. Expected numbers

| | Base | Seed 0 | Seed 1 |
|---|---|---|---|
| Held-out A (500 × 4) | 0.480 | 0.621 (+14.1 [11.4, 16.9]) | 0.613 (+13.3 [10.5, 16.1]) |
| Held-out B, lcbv5 (279 × 4) | 0.609 | 0.690 (+8.1 [4.9, 11.3]) | 0.711 (+10.1 [7.0, 13.4]) |
| Codeforces (408 × 2) | 0.212 | 0.412 | 0.395 |
| HumanEval+ / MBPP+ | 0.943 / 0.813 | 0.955 / 0.807 | 0.943 / 0.812 |
| IFEval / MMLU-Pro / MATH-500 (300, greedy) | 0.880 / 0.740 / 0.950 | 0.890 / 0.773 / 0.937 | 0.893 / 0.777 / 0.943 |

Sampling is at temperature 1, so reruns of the same checkpoint differ by about ±0.5–1 point on
held-out A (measured: 0.480 vs 0.485 base, 0.621 vs 0.616 seed 0).

## 7. Evaluate the released adapters (no training)

The [HF repo](https://huggingface.co/ik6626/Qwen3.8-27B-deepcoder-RL) holds standard PEFT adapters: seed 0 final at the root, `seed1/`, and `step50/seed0`,
`step50/seed1` (for another one, download its folder and pass `--src "$D/seed1"`). To evaluate one through this server:

```bash
source scripts/env.sh
D=$(hf download ik6626/Qwen3.8-27B-deepcoder-RL --include "adapter_*" "training_info.json")   # seed 0 final, 862 MB; prints the cache path
PORT=8100 RUN=demo scripts/run.sh server profiles/node8x-qwen38-27b-ctx32k.yaml   # in tmux; GPUs 0-2
.venv/bin/python scripts/import_adapter.py --profile profiles/node8x-qwen38-27b-ctx32k.yaml --name hf_seed0 --src "$D"
#   -> prints tinker://imported/sampler_weights/hf_seed0
scripts/eval_queue.sh 8100 runs/hf_evals hf_seed0 tinker://imported/sampler_weights/hf_seed0
```

`scripts/export_hf.py` is the reverse: it writes a store checkpoint as a PEFT directory.

**Serving with stock SGLang** (as on the model card): convert once, then start SGLang with `--lora-paths`.
`scripts/check_adapter.py` compares a stock SGLang server against this repo's server on the same tokens:

```bash
.venv/bin/python scripts/convert_to_sglang.py "$D" runs/adapter-sglang
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m sglang.launch_server --model-path Qwen/Qwen3.8-27B --enable-lora \
    --lora-paths deepcoder=runs/adapter-sglang --lora-target-modules all --max-lora-rank 32 --port 30400   # in tmux
TINKER_API_KEY=$ZROFORM_API_KEYS $ZROFORM_CLIENT_VENV/bin/python scripts/check_adapter.py \
    --sampler-path tinker://imported/sampler_weights/hf_seed0 --sglang-url http://127.0.0.1:30400 --lora-name deepcoder
```

Measured on 4 held-out problems (7,656 answer tokens): mean |logprob difference| 0.0074 nats/token between
the two servers with the adapter, the bf16 noise floor (the two servers' base models differ by 0.0072 on a
separate 300-token check), against 0.049 between adapter and base. So stock SGLang with the converted adapter computes the same model as every evaluation here.
The same adapter through this repo's server scored 0.570 on the first 100 held-out A problems (base 0.480; the
recorded seed-0 runs 0.588 and 0.593 on those problems): `RESULTS.md`, release checks.
