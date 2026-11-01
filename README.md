# Qwen3.8-27B DeepCoder RL

LoRA reinforcement learning on Qwen3.8-27B for competitive programming, with a reward the generated programs
can't fake, and an evaluation designed to tell a real held-out gain from noise.

**Adapters:** [ik6626/Qwen3.8-27B-deepcoder-RL](https://huggingface.co/ik6626/Qwen3.8-27B-deepcoder-RL)
(PEFT; usage with transformers and SGLang on the model card). **Write-up:** [docs/PAPER.md](docs/PAPER.md).
**Full results:** [RESULTS.md](RESULTS.md).

## Result

LoRA rank 32, 100 steps × 50 problems × 4 rollouts, two seeds, single-turn coding. Pass@1 over 4 samples at
temperature 1; gain in points with a paired bootstrap 95% CI over problems.

| Held-out set | Base | Seed 0 | Seed 1 |
|---|---|---|---|
| A: 500 DeepCoder problems, held out by problem (near-duplicates removed from training) | 0.480 | **0.621** (+14.1 [11.4, 16.9]) | **0.613** (+13.3 [10.5, 16.1]) |
| B: LiveCodeBench v5 test (279) | 0.609 | **0.690** (+8.1 [4.9, 11.3]) | **0.711** (+10.1 [7.0, 13.4]) |
| Codeforces (408; median 1 test per problem, weakly graded) | 0.212 | 0.412 | 0.395 |

- No regression beyond 2 points on HumanEval+, MBPP+, IFEval, MMLU-Pro or MATH-500.
- About ¾ of the gain is better solutions (pass rate of answers that finish: 0.597 → 0.706), about ¼ fewer
  answers running into the 12,288-token limit.
- The base score reproduces on the adapters' server (0.485 vs 0.480); no reward-hacking patterns in ~24k
  rewarded training rollouts.

## Use the adapter

See the [model card](https://huggingface.co/ik6626/Qwen3.8-27B-deepcoder-RL): transformers + PEFT directly, or
SGLang after a one-off, exact conversion (`convert_to_sglang.py`, in the HF repo and in `scripts/`).

## How it was trained

**Zro-form** (`zroform/`) is a training server for the Tinker wire protocol, so the unmodified Tinker SDK and
tinker-cookbook recipes run against it on your own GPUs:

- **trainer:** Hugging Face transformers + PEFT LoRA on one GPU, with the Tinker loss functions;
- **sampler:** a stock SGLang server; after each `save_weights_for_sampler`, the new adapter is handed over as
  in-memory tensors (CUDA IPC);
- **engine:** sessions, per-model operation ordering, retry de-duplication, and a checkpoint store with resume.

The cookbook's `code_rl` recipe runs on it unchanged except for two things (`scripts/code_rl.py`):
- the grader runs each program in a fresh locked-down container and compares outputs outside it
  (`scripts/safe_grader.py`), because the stock grader scores a program that calls `os._exit(0)` as a pass;
- single turn, no tool, because the recipe's `check_solution` tool returns the tests' expected outputs.

## Reproduce

[REPRODUCE.md](REPRODUCE.md): environment, data, the grader self-tests, training (≈63 h on 2 × 3 GPUs),
evaluation, expected numbers, and evaluating the released adapters without retraining.

Requirements: NVIDIA GPUs with ≥ 96 GB each (3 per seed; the run used 8× RTX PRO 6000 Blackwell), Docker,
Python 3.12 with [`uv`](https://docs.astral.sh/uv/). Machine-specific paths go in `.env` (gitignored; see
[.env.example](.env.example)); everything defaults to paths inside the repo.

## Layout

| Path | What |
|---|---|
| `zroform/` | the server: Tinker wire protocol (`api.py`, `wire.py`), engine, trainer, SGLang sampler, checkpoint store |
| `profiles/` | the three server profiles of the run (seed 0, seed 1, evaluation) |
| `scripts/` | `run.sh` (start a server), `main_run.sh` (training launcher with supervisor), `code_rl.py` (the recipe wrapper), `safe_grader.py` + `runner.Dockerfile`, `split.py`, evaluations (`eval_code.py`, `eval_regress.py`, `eval_bench.py`, the two `eval_queue*.sh`), `health_report.py`, self-tests, adapter tools (`export_hf.py`, `import_adapter.py`, `convert_to_sglang.py`, `check_adapter.py`), `env.sh` |
| `data/split_manifest.json` | the held-out / validation / training split (problem hashes, no texts) |
| `requirements-client.txt` | the client venv's exact pins (tinker SDK, cookbook) |
| `tests/` | CPU tests (the real Tinker SDK against a tiny model): `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests -q` |

## Licence

Apache-2.0 ([LICENSE](LICENSE)). The base model (Qwen3.8-27B) is Apache-2.0; the training data
(DeepCoder-Preview-Dataset) is MIT.
