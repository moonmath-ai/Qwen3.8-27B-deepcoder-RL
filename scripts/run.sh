#!/usr/bin/env bash
# Start the training server.
# Usage: run.sh server [profile]          (default profiles/node8x-qwen38-27b-ctx32k.yaml)
# Env: RUN (output dir name), PORT (default 8100). Machine paths: scripts/env.sh (<repo>/.env).
set -euo pipefail
source "$(dirname "$0")/env.sh"
REPO=$ZF_REPO                     # server venv: $REPO/.venv (uv sync)
PORT=${PORT:-8100}
RUN=${RUN:-zroform_$(date +%Y%m%d_%H%M)}
OUT=$ZROFORM_OUT_ROOT/$RUN
export TMPDIR=$ZROFORM_TMP/rt
mkdir -p "$TMPDIR" "$OUT"

case "${1:?usage: run.sh server [profile]}" in
  server)
    cd "$REPO"
    # SGLang's JIT kernels call `ninja` by bare name: the venv's bin/ must be on PATH (else the
    # scheduler dies during CUDA-graph capture with FileNotFoundError: 'ninja').
    export PATH="$REPO/.venv/bin:$PATH"
    export ZROFORM_PROFILE=${2:-profiles/node8x-qwen38-27b-ctx32k.yaml}
    # zroform.launch hides the GPUs the profile does not use before CUDA initialises.
    exec .venv/bin/python -m zroform.launch --host 127.0.0.1 --port "$PORT" ;;
esac
