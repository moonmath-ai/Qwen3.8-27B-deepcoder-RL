#!/usr/bin/env bash
# Regression eval queue for one model (RESULTS.md): HumanEval+ x2 and MBPP+ x2 through the safe
# grader (eval_regress.py), then IFEval, MMLU-Pro, MATH-500 (300 each, greedy) via eval_bench.py.
#   scripts/eval_queue_regress.sh <port> <out_dir> <label> [sampler_path]   (no sampler_path = the base model)
set -uo pipefail
port=$1 out=$2 label=$3 sp=${4:-}
source "$(dirname "$0")/env.sh"
# Once the model and data are cached, set HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 (e.g. in .env): offline avoids the
# Hub rate limit (429) when several evals start at once.
# eval_bench is the exception: the cookbook's load_dataset(<repo>) cannot resolve hub-cache-only datasets offline.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}
cd "$ZF_REPO"
export TINKER_API_KEY=${ZROFORM_API_KEYS%%,*} TINKER_TELEMETRY=0 ZF_GRADER_CONCURRENCY=48
for b in humanevalplus mbppplus; do
  echo "[$(date -Is)] $label: $b x2"
  zf_docker "$ZROFORM_CLIENT_VENV/bin/python scripts/eval_regress.py run --bench $b --n 2 ${sp:+--sampler-path $sp} \
      --base-url http://127.0.0.1:$port --out $out/${label}_$b" || echo "[$(date -Is)] $label: $b FAILED"
done
echo "[$(date -Is)] $label: ifeval, mmlu_pro, math500 (300 each, greedy)"
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 "$ZROFORM_CLIENT_VENV/bin/python" scripts/eval_bench.py --base-url "http://127.0.0.1:$port" --model Qwen/Qwen3.8-27B \
    --renderer qwen3_8_low_reasoning --max-tokens 12288 --max-examples 300 --bench ifeval mmlu_pro math500 \
    ${sp:+--adapter $label=$sp --skip-base} --out "$out/${label}_bench/bench.json" || echo "[$(date -Is)] $label: eval_bench FAILED"
echo "[$(date -Is)] $label: regression queue done"
touch "$out/$label.regress.done"
