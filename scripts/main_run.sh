#!/usr/bin/env bash
# Qwen3.8-27B DeepCoder RL, 100 steps x 2 seeds, plus held-out evals (RESULTS.md).
#   scripts/main_run.sh start    servers, both seeds, base evals, supervisor (all in tmux; refuses if GPUs are busy)
#   scripts/main_run.sh stop     clean stop of everything (loses at most the step in progress)
#   scripts/main_run.sh resume   same as start; runs continue from their last (rolling) checkpoint
#   scripts/main_run.sh status   steps, rewards, validation, checkpoints, sessions, GPUs, supervisor log
# Layout: seed 0 = trainer GPU 0 + SGLang GPUs 1-2 (port 8100); seed 1 = GPU 4 + GPUs 5-6 (8200);
# eval server = trainer GPU 3 + SGLang GPU 7 (8300). The supervisor (tmux zf-main-sup) restarts and resumes
# crashed or hung runs on its own, and starts each seed's adapter evals on that seed's server when it finishes.
set -uo pipefail
source "$(dirname "$0")/env.sh"
REPO=$ZF_REPO
RUN_NAME=${MAIN_RUN_NAME:-main}
ROOT=$ZROFORM_OUT_ROOT/$RUN_NAME
MAX_STEPS=${MAIN_MAX_STEPS:-100}
PROFILE_0=profiles/node8x-qwen38-27b-ctx32k.yaml;      PORT_0=8100; SPORT_0=30100; GPUS_0="0,1,2"
PROFILE_1=profiles/node8x-qwen38-27b-ctx32k-b.yaml;    PORT_1=8200; SPORT_1=30200; GPUS_1="4,5,6"
PROFILE_E=profiles/node8x-qwen38-27b-ctx32k-eval.yaml; PORT_E=8300; SPORT_E=30300; GPUS_E="3,7"
mkdir -p "$ROOT/evals"
cd "$REPO"
log() { echo "[$(date -Is)] $*" | tee -a "$ROOT/main_run.log"; }

env_common() {  # the client environment, as a string for tmux
  echo "source $REPO/scripts/env.sh; cd $REPO; \
export TINKER_API_KEY=\${ZROFORM_API_KEYS%%,*} TINKER_TELEMETRY=0 ZF_GRADER_CONCURRENCY=48 \
ZF_SPLIT_MANIFEST=$REPO/data/split_manifest.json ZF_ROLLING_SAVE_EVERY=1 ZF_ROLLING_TTL_SECONDS=604800 ZF_CKPT_TTL_SECONDS=none"
}
healthy() { curl -s -m 5 -o /dev/null "http://127.0.0.1:$1/api/v1/healthz"; }
gpus_free() { for g in ${1//,/ }; do [ "$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] || return 1; done; }

start_server() {  # $1 = tag (0|1|E)
  local p port gpus; eval "p=\$PROFILE_$1 port=\$PORT_$1 gpus=\$GPUS_$1"
  healthy "$port" && return 0
  tmux kill-session -t "zf-main-srv$1" 2>/dev/null
  for i in $(seq 1 180); do gpus_free "$gpus" && break; sleep 2; done
  gpus_free "$gpus" || { log "server $1: GPUs $gpus are not free; not starting"; return 1; }
  tmux new-session -d -s "zf-main-srv$1" "cd $REPO && PORT=$port RUN=$RUN_NAME/server$1 scripts/run.sh server $p >> $ROOT/server$1.log 2>&1"
  for i in $(seq 1 600); do healthy "$port" && { log "server $1 up on $port"; return 0; }; tmux has-session -t "zf-main-srv$1" 2>/dev/null || break; sleep 2; done
  log "server $1 failed to start (see $ROOT/server$1.log)"; return 1
}
stop_server() {  # also kills a crashed server's leftovers (its trainer / SGLang processes, found by their ports)
  local port sport; eval "port=\$PORT_$1 sport=\$SPORT_$1"
  tmux send-keys -t "zf-main-srv$1" C-c 2>/dev/null
  for i in $(seq 1 120); do tmux has-session -t "zf-main-srv$1" 2>/dev/null || break; sleep 1; done
  tmux kill-session -t "zf-main-srv$1" 2>/dev/null
  pkill -u "$USER" -f "zroform.launch --host 127.0.0.1 --port $port" 2>/dev/null
  pkill -u "$USER" -f "sglang.launch_server.*--port $sport" 2>/dev/null
  sleep 5
}
seed_done() { grep -q '"name": "final"' "$ROOT/seed$1/checkpoints.jsonl" 2>/dev/null; }
start_client() {  # $1 = seed
  local port; eval "port=\$PORT_$1"
  seed_done "$1" && return 0
  tmux kill-session -t "zf-main-cli$1" 2>/dev/null
  tmux new-session -d -s "zf-main-cli$1" "$(env_common); zf_docker \"$ZROFORM_CLIENT_VENV/bin/python scripts/code_rl.py \
base_url=http://127.0.0.1:$port model_name=Qwen/Qwen3.8-27B log_path=$ROOT/seed$1 renderer_name=qwen3_8_low_reasoning \
max_tokens=12288 group_size=4 groups_per_batch=50 max_steps=$MAX_STEPS eval_every=10 save_every=10 seed=$1 \
behavior_if_log_dir_exists=resume\" >> $ROOT/client$1.log 2>&1"
  log "client $1 started (resumes from its last checkpoint if any)"
}
stop_client() {  # every process in the client's tmux pane session (sg -> shell -> python): SIGTERM, then SIGKILL
  local pane sid; pane=$(tmux list-panes -t "zf-main-cli$1" -F '#{pane_pid}' 2>/dev/null) || return 0
  sid=$(ps -o sid= -p "$pane" | tr -d ' ')
  [ -n "$sid" ] && pkill -TERM -s "$sid" 2>/dev/null; sleep 10; [ -n "$sid" ] && pkill -KILL -s "$sid" 2>/dev/null
  tmux kill-session -t "zf-main-cli$1" 2>/dev/null
}
eval_queue() {  # $1 = tag, $2 = port, $3 = sampler path or "" (base), $4 = label
  tmux kill-session -t "zf-main-eval$1" 2>/dev/null
  tmux new-session -d -s "zf-main-eval$1" "$REPO/scripts/eval_queue.sh $2 $ROOT/evals $4 $3 >> $ROOT/evals/$4.log 2>&1"
  log "eval queue $4 started on port $2"
}
final_sampler() { python3 -I -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1]) if json.loads(l).get('name')=='final']; print(r[-1]['sampler_path'])" "$ROOT/seed$1/checkpoints.jsonl"; }

supervise() {  # loop: restart/resume crashed or hung seeds; start adapter evals when a seed finishes
  declare -A restarts=() recent=()
  log "supervisor up"
  while true; do
    # Disk guard: other users share this disk. Warn under 50 GB free; under 20 GB stop cleanly (resumable)
    # rather than risk a half-written checkpoint.
    free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
    if [ "$free_gb" -lt 20 ]; then
      log "DISK: only ${free_gb} GB free under $ROOT -> stopping everything cleanly (resume with: main_run.sh resume)"
      # detached: `stop` kills this supervisor's tmux session, which would otherwise kill the stop itself
      setsid nohup "$REPO/scripts/main_run.sh" stop >> "$ROOT/main_run.log" 2>&1 < /dev/null & exit 0
    elif [ "$free_gb" -lt 50 ] && [ ! -e "$ROOT/disk.warned" ]; then
      log "DISK: ${free_gb} GB free under $ROOT (warning; stops at 20 GB)"; touch "$ROOT/disk.warned"
    fi
    for s in 0 1; do
      local port; eval "port=\$PORT_$s"
      if seed_done "$s"; then
        if [ ! -e "$ROOT/evals/adapter_seed$s.started" ]; then
          touch "$ROOT/evals/adapter_seed$s.started"; stop_client "$s"
          healthy "$port" || start_server "$s"
          eval_queue "A$s" "$port" "$(final_sampler "$s")" "adapter_seed$s"
        fi
        continue
      fi
      local reason=""
      tmux has-session -t "zf-main-cli$s" 2>/dev/null || reason="client exited"
      healthy "$port" || reason="server unhealthy"
      if [ -z "$reason" ] && [ -z "$(find "$ROOT/client$s.log" -newermt '-120 minutes' 2>/dev/null)" ]; then reason="no progress for 120 min"; fi
      [ -z "$reason" ] && continue
      now=$(date +%s); recent[$s]="$(for t in ${recent[$s]:-}; do [ $((now - t)) -lt 3600 ] && echo -n "$t "; done)"
      if [ "${restarts[$s]:-0}" -ge 12 ] || [ "$(echo ${recent[$s]} | wc -w)" -ge 3 ]; then
        [ -e "$ROOT/seed$s.GAVE_UP" ] || { log "seed $s: GAVE UP after ${restarts[$s]:-0} restarts ($reason); needs a human"; touch "$ROOT/seed$s.GAVE_UP"; }
        continue
      fi
      restarts[$s]=$(( ${restarts[$s]:-0} + 1 )); recent[$s]="${recent[$s]} $now"
      log "seed $s: $reason -> restart #${restarts[$s]}"
      stop_client "$s"
      if [ "$reason" = "client exited" ]; then
        sleep 75  # the dead client's session holds the training slot until the server's 60 s idle reaper
      else
        stop_server "$s"
      fi
      start_server "$s" && start_client "$s"
    done
    sleep 60
  done
}

status() {
  for s in 0 1; do
    echo "== seed $s: client $(tmux has-session -t zf-main-cli$s 2>/dev/null && echo running || echo stopped), server $(healthy $(eval echo \$PORT_$s) && echo healthy || echo down)$(seed_done $s && echo ', FINISHED')$([ -e $ROOT/seed$s.GAVE_UP ] && echo ', GAVE UP')"
    [ -s "$ROOT/seed$s/metrics.jsonl" ] && python3 -I - "$ROOT/seed$s/metrics.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
tr = [r for r in rows if "env/all/reward/total" in r]
va = [r for r in rows if any(k.startswith("test/") and k.endswith("reward/total") for k in r)]
if tr:
    last = tr[-1]
    print(f"  step {last['step']}: reward {last['env/all/reward/total']:.3f}, mixed groups {last.get('env/all/by_group/frac_mixed', float('nan')):.2f}, "
          f"step time {last.get('time/total', 0) / 60:.1f} min; reward last 10 steps {sum(r['env/all/reward/total'] for r in tr[-10:]) / len(tr[-10:]):.3f}")
for r in va[-3:]:
    k = [k for k in r if k.startswith("test/") and k.endswith("reward/total")][0]
    print(f"  validation @ step {r['step']}: {r[k]:.3f}")
PY
    tail -1 "$ROOT/seed$s/checkpoints.jsonl" 2>/dev/null | cut -c1-120 | sed 's/^/  last checkpoint: /'
  done
  echo "== evals: $(ls $ROOT/evals 2>/dev/null | grep -E '\.done$' | tr '\n' ' ')"
  for d in "$ROOT"/evals/*/; do [ -s "$d/results.jsonl" ] && echo "  $(basename $d): $(wc -l < $d/results.jsonl) problems done"; done
  echo "== tmux: $(tmux ls 2>/dev/null | grep -oE '^zf-main-[a-zA-Z0-9]+' | tr '\n' ' ')"
  echo "== GPUs (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
  echo "== supervisor log:"; tail -5 "$ROOT/main_run.log" 2>/dev/null | sed 's/^/  /'
}

case "${1:?start|stop|resume|status|supervise}" in
  start|resume)
    for t in 0 1 E; do start_server "$t" || exit 1; done
    start_client 0; start_client 1
    if [ ! -e "$ROOT/evals/base.done" ] && ! tmux has-session -t zf-main-evalB 2>/dev/null; then
      eval_queue B "$PORT_E" "" base
    fi
    tmux has-session -t zf-main-sup 2>/dev/null || tmux new-session -d -s zf-main-sup "$REPO/scripts/main_run.sh supervise"
    log "$1: done"; status ;;
  supervise) supervise ;;
  stop)
    tmux kill-session -t zf-main-sup 2>/dev/null; log "supervisor stopped"
    for s in 0 1; do stop_client "$s"; done
    for t in B A0 A1; do tmux kill-session -t "zf-main-eval$t" 2>/dev/null; done
    for t in 0 1 E; do stop_server "$t"; done
    ids=$(zf_docker 'docker ps -q --filter name=zf-run-'); [ -n "$ids" ] && zf_docker "docker kill $ids" >/dev/null
    log "stopped (runs resume from their last checkpoint; evals resume per problem)"; status ;;
  status) status ;;
esac
