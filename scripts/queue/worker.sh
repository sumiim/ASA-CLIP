#!/usr/bin/env bash
# At most MAX_JOBS_TOTAL of *our* main.py across BOTH GPUs (default 2, not per-GPU).
# Launch only if the chosen GPU has free >= MIN_FREE_MB (7 GB).
# Count via `ps` (parent python -u main.py), not nvidia-smi: CUDA_VISIBLE_DEVICES
# jobs are invisible to nvidia-smi until they allocate, which over-pops.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QDIR="$ROOT/scripts/queue"
PYTHON="${PYTHON:-python}"
USER_NAME="${USER:?USER must be set}"
MAX_JOBS_TOTAL="${MAX_JOBS_TOTAL:-2}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-7168}"
INTERVAL="${INTERVAL:-20}"
LOG="$ROOT/revision_outputs/queue/worker.log"
LAUNCH_LOCK="$QDIR/.launch.lock"

mkdir -p "$(dirname "$LOG")"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
log "queue worker start max_jobs_total=${MAX_JOBS_TOTAL} max_jobs_per_gpu=${MAX_JOBS_PER_GPU} min_free_mb=${MIN_FREE_MB}"

# Parent python -u main.py owned by us. Optional gpu filter uses CUDA_VISIBLE_DEVICES.
# Visible as soon as exec happens, including during dataloader / CLIP load.
our_job_count() {
  local gpu="${1:-}"
  local n=0 pid ppid cmd pcmd cuda
  while read -r pid ppid cmd; do
    [ -n "${pid:-}" ] || continue
    # Real launches are `.../python -u main.py --device ...`.
    # Do not match diagnostic bash/awk whose command line only mentions that string.
    case "$cmd" in
      *'/python -u main.py --'*|*'python -u main.py --'*) ;;
      *) continue ;;
    esac
    pcmd="$(ps -o cmd= -p "$ppid" 2>/dev/null || true)"
    case "$pcmd" in
      *'/python -u main.py --'*|*'python -u main.py --'*) continue ;;
    esac
    if [ -n "$gpu" ]; then
      cuda="$(tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | awk -F= '/^CUDA_VISIBLE_DEVICES=/{print $2; exit}')"
      [ "${cuda:-}" = "$gpu" ] || continue
    fi
    n=$((n + 1))
  done < <(ps -u "$USER_NAME" -o pid=,ppid=,cmd=)
  echo "$n"
}

free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1)}'
}

pick_gpu() {
  # One of our jobs per GPU. Prefer more free memory. Empty if none has 7GB.
  local best="" best_free=-1
  local gpu n free
  for gpu in 0 1; do
    n="$(our_job_count "$gpu")"
    free="$(free_mb "$gpu")"
    if [ "$n" -ge "$MAX_JOBS_PER_GPU" ]; then
      continue
    fi
    if [ "$free" -lt "$MIN_FREE_MB" ]; then
      continue
    fi
    if [ "$free" -gt "$best_free" ]; then
      best="$gpu"
      best_free="$free"
    fi
  done
  echo "$best"
}

while true; do
  n0="$(our_job_count 0)"
  n1="$(our_job_count 1)"
  total=$((n0 + n1))
  if [ "$total" -ge "$MAX_JOBS_TOTAL" ]; then
    sleep "$INTERVAL"
    continue
  fi
  gpu="$(pick_gpu)"
  if [ -z "$gpu" ]; then
    sleep "$INTERVAL"
    continue
  fi
  # Serialize pop+launch so two loops cannot over-pop while python is starting.
  exec 9>"$LAUNCH_LOCK"
  if ! flock -n 9; then
    sleep "$INTERVAL"
    continue
  fi
  total="$(our_job_count)"
  if [ "$total" -ge "$MAX_JOBS_TOTAL" ]; then
    flock -u 9
    sleep "$INTERVAL"
    continue
  fi
  gpu="$(pick_gpu)"
  if [ -z "$gpu" ]; then
    flock -u 9
    sleep "$INTERVAL"
    continue
  fi
  job="$("$PYTHON" "$QDIR/pop_job.py" || true)"
  if [ -z "${job:-}" ]; then
    log "queue empty total=${total}"
    flock -u 9
    sleep "$INTERVAL"
    continue
  fi
  free="$(free_mb "$gpu")"
  log "gpu=${gpu} ours=${n0}+${n1}=${total} free=${free}MB launch ${job}"
  screen -dmS "asa_${job}" bash -lc \
    "cd '$ROOT' && '$QDIR/launch_job.sh' ${job} ${gpu}; echo JOB_EXIT:\$?; exec bash"
  # Wait until ps sees the parent AND this GPU's free mem dropped (allocation),
  # so the next loop cannot stack a second job on the same leftover 7GB.
  seen=0
  allocated=0
  for _ in $(seq 1 45); do
    sleep 2
    now="$(our_job_count)"
    if [ "$now" -gt "$total" ]; then
      seen=1
    fi
    now_free="$(free_mb "$gpu")"
    if [ "$now_free" -le $((free - 1500)) ]; then
      allocated=1
    fi
    if [ "$seen" -eq 1 ] && [ "$allocated" -eq 1 ]; then
      break
    fi
  done
  if [ "$seen" -eq 0 ]; then
    log "warn: ${job} not visible in ps after 90s; holding lock to avoid over-pop"
    sleep 30
  elif [ "$allocated" -eq 0 ]; then
    log "warn: ${job} visible but gpu=${gpu} free did not drop 1.5GB (still $(free_mb "$gpu")MB); not stacking"
    sleep 20
  fi
  flock -u 9
done
