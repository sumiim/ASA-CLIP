#!/usr/bin/env bash
# Four-direction semantic-anchor ablations under the independent-val protocol.
# Seed is fixed at 42. Set SKIP_ASL=1 to reuse the main-table seed-42 ASL run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_PATH="${DATA_PATH:-../data/}"
OUTPUT_DIR="${OUTPUT_DIR:-./revision_outputs/anchor_ablations}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-42}"
SKIP_ASL="${SKIP_ASL:-0}"

DATASETS=(sketchy tuberlin quickdraw)

# name|sketch_text_target|photo_loss_target|use_text_adapter
VARIANTS=(
  "te|raw|raw|False"
  "ta|adapted|adapted|True"
  "asl|adapted|raw|True"
  "raw_adapted|raw|adapted|True"
)

validation_metric_for() {
  case "$1" in
    sketchy) echo "map@200" ;;
    *) echo "map@all" ;;
  esac
}

echo "Launching four-direction anchor ablations"
echo "  ROOT=$ROOT DEVICE=$DEVICE SEED=$SEED SKIP_ASL=$SKIP_ASL"
echo "One GPU per job. Check nvidia-smi (>=16 GB free) before starting."

for dataset in "${DATASETS[@]}"; do
  metric="$(validation_metric_for "$dataset")"
  for variant in "${VARIANTS[@]}"; do
    IFS='|' read -r name sketch_tgt photo_tgt adapter <<< "$variant"
    if [[ "$name" == "asl" && "$SKIP_ASL" == "1" ]]; then
      echo "Skip $dataset $name (reuse main-table seed=$SEED)"
      continue
    fi
    exp_name="${dataset}_${name}_lambda0.7_db512_seed${SEED}"
    echo "============================================================"
    echo "RUN dataset=$dataset variant=$name sketch=$sketch_tgt photo=$photo_tgt adapter=$adapter"
    echo "============================================================"
    "$PYTHON" main.py \
      --data_path "$DATA_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --device "$DEVICE" \
      --dataset "$dataset" \
      --seed "$SEED" \
      --batch_size 32 \
      --num_workers 8 \
      --pin_memory True \
      --epochs 400 \
      --patience 10 \
      --lr 1e-5 \
      --weight_decay 0.05 \
      --validation_metric "$metric" \
      --eval_metrics map@all map@200 p@200 p@100 \
      --train_sketch True \
      --train_photo True \
      --use_semantic_alignment True \
      --use_text_adapter "$adapter" \
      --text_bottleneck 512 \
      --lambda2 0.7 \
      --sketch_text_target "$sketch_tgt" \
      --photo_loss_target "$photo_tgt" \
      --eval_test_at_end True \
      --exp_name "$exp_name"
  done
done
