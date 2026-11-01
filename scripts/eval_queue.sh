#!/usr/bin/env bash
# Held-out eval queue for one model (RESULTS.md): heldout_a x4, heldout_b x4, codeforces x2, in order.
#   scripts/eval_queue.sh <port> <out_dir> <label> [sampler_path]   (no sampler_path = the base model)
# eval_code.py resumes per problem, so rerunning a stopped queue continues where it left off.
set -uo pipefail
port=$1 out=$2 label=$3 sp=${4:-}
source "$(dirname "$0")/env.sh"
# Once the model and data are cached, set HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 (e.g. in .env): offline avoids the
# Hub rate limit (429) when several evals start at once.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}
cd "$ZF_REPO"
export TINKER_API_KEY=${ZROFORM_API_KEYS%%,*} TINKER_TELEMETRY=0 ZF_GRADER_CONCURRENCY=48
for spec in heldout_a:4 heldout_b:4 codeforces:2; do
  set_name=${spec%%:*} n=${spec##*:}
  echo "[$(date -Is)] $label: $set_name x$n"
  zf_docker "$ZROFORM_CLIENT_VENV/bin/python scripts/eval_code.py run --set $set_name --n $n \
      ${sp:+--sampler-path $sp} --base-url http://127.0.0.1:$port --out $out/${label}_$set_name" || echo "[$(date -Is)] $label: $set_name FAILED"
done
echo "[$(date -Is)] $label: queue done"
touch "$out/$label.done"
