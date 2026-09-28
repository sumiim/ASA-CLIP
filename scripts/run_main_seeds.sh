#!/usr/bin/env bash
# Main-table ASA-CLIP runs: 3 datasets x 3 seeds.
# One GPU per job. Do not launch until free memory on that GPU is >= 16 GB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_PATH="${DATA_PATH:-../data/}"
OUTPUT_DIR="${OUTPUT_DIR:-./revision_outputs/main_seeds}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"

SEEDS=(42 43 44)
DATASETS=(sketchy tuberlin quickdraw)

common_flags=(
  --data_path "$DATA_PATH"
  --output_dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --batch_size 32
  --num_workers 8
  --pin_memory True
  --epochs 400
  --patience 10
  --lr 1e-5
  --weight_decay 0.05
  --eval_metrics map@all map@200 p@200 p@100
  --train_sketch True
  --train_photo True
  --use_semantic_alignment True
  --use_text_adapter True
  --text_bottleneck 512
  --lambda2 0.7
  --sketch_text_target adapted
  --photo_loss_target raw
  --eval_test_at_end True
)

validation_metric_for() {
  case "$1" in
    sketchy) echo "map@200" ;;
    *) echo "map@all" ;;
  esac
}

echo "Launching ASA-CLIP main-table runs"
echo "  ROOT=$ROOT"
echo "  DEVICE=$DEVICE"
echo "  DATA_PATH=$DATA_PATH"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  SEEDS=${SEEDS[*]}"
echo "One GPU per job. Check nvidia-smi (>=16 GB free) before starting."

for dataset in "${DATASETS[@]}"; do
  metric="$(validation_metric_for "$dataset")"
  for seed in "${SEEDS[@]}"; do
    exp_name="${dataset}_asl_lambda0.7_db512_seed${seed}"
    echo "============================================================"
    echo "RUN dataset=$dataset seed=$seed metric=$metric"
    echo "============================================================"
    "$PYTHON" main.py \
      "${common_flags[@]}" \
      --dataset "$dataset" \
      --seed "$seed" \
      --validation_metric "$metric" \
      --exp_name "$exp_name"
  done
done
