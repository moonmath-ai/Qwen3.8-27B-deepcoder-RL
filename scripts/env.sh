# Shared environment for the shell scripts: `source scripts/env.sh` (from any directory).
# Machine-specific paths go in <repo>/.env, which is gitignored: copy .env.example and edit it.
# Every variable has a default inside the repo, so an empty .env works too.
ZF_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -f "$ZF_REPO/.env" ]; then set -a; source "$ZF_REPO/.env"; set +a; fi
export ZROFORM_OUT_ROOT=${ZROFORM_OUT_ROOT:-$ZF_REPO/runs}        # run outputs: rollouts, metrics, evals, logs
export ZROFORM_DATA=${ZROFORM_DATA:-$ZROFORM_OUT_ROOT}            # profiles' checkpoint store root (store/ under it)
export ZROFORM_CLIENT_VENV=${ZROFORM_CLIENT_VENV:-$ZF_REPO/.venv-client}  # tinker SDK + cookbook (REPRODUCE.md §1)
export ZROFORM_TMP=${ZROFORM_TMP:-$ZROFORM_OUT_ROOT/tmp}           # scratch space
# The model and datasets are ungated: keep tokens out of child processes and logs.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
# Local dev API key of our own server (not a secret; every client of this server uses it).
export ZROFORM_API_KEYS=${ZROFORM_API_KEYS:-tml-zroform-dev}

# Run a command string with access to the Docker daemon: directly, or through `sg docker` when the user
# joined the docker group after this shell started. `zf_docker "cmd"`, or `setsid $ZF_DOCKER_RUN "cmd"`
# where a program (not a shell function) is needed.
if docker info >/dev/null 2>&1; then ZF_DOCKER_RUN="bash -c"; else ZF_DOCKER_RUN="sg docker -c"; fi
zf_docker() { $ZF_DOCKER_RUN "$1"; }
