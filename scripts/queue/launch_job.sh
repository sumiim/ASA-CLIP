#!/usr/bin/env bash
# Read scripts/queue/jobs/<id>.conf and run main.py on CUDA_VISIBLE_DEVICES=$GPU.
set -euo pipefail
JOB_ID="${1:?job id}"
GPU="${2:?physical gpu}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONF="$ROOT/scripts/queue/jobs/${JOB_ID}.conf"
if [ ! -f "$CONF" ]; then
  echo "missing conf $CONF"
  exit 1
fi
# shellcheck disable=SC1090
source "$CONF"

# Defaults so older ASL/anchor confs keep working.
SINGLE_BRANCH="${SINGLE_BRANCH:-False}"
TRAIN_SKETCH="${TRAIN_SKETCH:-True}"
TRAIN_PHOTO="${TRAIN_PHOTO:-True}"
SEMANTIC="${SEMANTIC:-True}"

PYTHON="${PYTHON:-python}"
DATA_PATH="${DATA_PATH:-../data/}"
cd "$ROOT"
mkdir -p "$OUTPUT_DIR"

EXP="${DATASET}_${VARIANT}_lambda${LAMBDA2}_db${BOTTLENECK}_seed${SEED}"
echo "launch ${JOB_ID} gpu=${GPU} exp=${EXP} out=${OUTPUT_DIR} $(date '+%Y-%m-%d %H:%M:%S')"

export CUDA_VISIBLE_DEVICES="$GPU"
exec "$PYTHON" -u main.py \
  --device cuda:0 \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --exp_name "$EXP" \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --batch_size "${BATCH}" \
  --num_workers 8 \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --lr 1e-5 \
  --weight_decay 0.05 \
  --validation_metric "$VALIDATION_METRIC" \
  --eval_metrics map@all map@200 p@200 p@100 \
  --use_single_branch "$SINGLE_BRANCH" \
  --train_sketch "$TRAIN_SKETCH" \
  --train_photo "$TRAIN_PHOTO" \
  --use_semantic_alignment "$SEMANTIC" \
  --use_text_adapter "$ADAPTER" \
  --text_bottleneck "$BOTTLENECK" \
  --lambda2 "$LAMBDA2" \
  --sketch_text_target "$SKETCH_TEXT" \
  --photo_loss_target "$PHOTO_TEXT" \
  --eval_test_at_end True \
  --method "${METHOD:-asa}" \
  --n_prompts "${N_PROMPTS:-3}" \
  --prompt_dim "${PROMPT_DIM:-768}" \
  --prompt_lr "${PROMPT_LR:-1e-4}"
