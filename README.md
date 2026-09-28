# ASA-CLIP

Code for **ASA-CLIP**: modality-independent LayerNorm-only PEFT on two visual CLIP encoders, plus asymmetric semantic anchors (ASL) for zero-shot sketch-based image retrieval (ZS-SBIR).

This package is the **revision-protocol** training code (independent seen-class validation). It is **not** the original submission snapshot that selected checkpoints on the unseen test split.

Paper-aligned training defaults used in the revised tables:

- `--lambda2 0.7`
- `--text_bottleneck 512`
- prompt template `a photo of a {class}`
- single-positive bidirectional InfoNCE
- official unseen test evaluated **once**, from the best **validation** checkpoint

Argparse defaults in `main.py` (`--lambda2 0.5`, `--text_bottleneck 256`) are **not** the paper runs. Always pass the flags below.

## Independent val / test protocol

Official unseen test classes are unchanged. A seen-class holdout is used only for early stopping and checkpoint selection. The holdout is drawn with `DATASET_SPLIT_SEED = 42` in `dataset.py` and does **not** follow `--seed`.

| Dataset | Official split | Train / val / test classes | Val metric |
|---|---|---|---|
| Sketchy-Ext | 104 seen / 21 unseen | 94 / 10 / 21 | `map@200` |
| TU-Berlin-Ext | 220 seen / 30 unseen | 200 / 20 / 30 | `map@all` |
| QuickDraw-Ext | 80 seen / 30 unseen | 72 / 8 / 30 | `map@all` |

Training loop:

1. Every epoch evaluates **val only**.
2. `best_model_by_<metric>.pth` is written only when val improves.
3. After early stopping (or max epochs), the best val checkpoint is loaded and the official unseen test split is scored **once**.
4. `evaluation_metrics.csv` has `split=val` rows during training and a single `split=test` row at the end.

Do not use test metrics for model selection.

## Installation

Python 3.9+ and a CUDA GPU are expected. The revision experiments used PyTorch 2.7.1 + CUDA 11.8 and `open_clip_torch==3.2.0`.

```bash
conda create -n asa-clip python=3.10 -y
conda activate asa-clip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

CPU-only can load the code, but training and the official retrieval eval need a GPU.

## CLIP weights

The visual backbone is OpenCLIP **ViT-B/16** with the OpenAI checkpoint.

Resolution order in `model.py`:

1. Environment variable `ASA_CLIP_WEIGHTS` pointing at a local OpenCLIP `pytorch_model.bin`
2. `--pretrained` if it is an existing file path
3. OpenCLIP tag, default `openai` (`open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")`)

Example with a local file (offline machines):

```bash
export ASA_CLIP_WEIGHTS=/path/to/openai_clip_ViT-B-16/pytorch_model.bin
```

Do not commit weight files. See `pretrained/README.md`.

## Data layout

Datasets are **not** included. Point `--data_path` at a directory that contains the three extended ZS-SBIR corpora:

```text
<data_path>/
  Sketchy_extended/
    sketch/tx_000000000000/<class>/*.png
    photo/tx_000000000000/<class>/*.jpg
    extended_photo/<class>/*.jpg
  TUBerlin/
    png_ready/<class>/*.png
    ImageResized_ready/<class>/*.jpg
  quickdraw_extended/
    sketches/<class>/*.png
    photos/<class>/*.jpg
```

If Sketchy has no `sketch/tx_000000000000` subdirectory, the loader falls back to `sketch/<class>/`.

Public sources (follow each dataset's license):

- Sketchy / Sketchy-Ext: [Sketchy dataset](https://sketchy.eye.gatech.edu/) and the standard SBIR extended-photo split used in ZS-SBIR papers
- TU-Berlin / TU-Berlin-Ext: TU-Berlin sketches + ImageNet-style photo extension used by ZS-SBIR
- QuickDraw-Ext: the QuickDraw-extended ZS-SBIR release used in recent CLIP-based SBIR papers

After placing the data, check that the independent split is stable:

```bash
python scripts/check_val_split.py /path/to/data
```

Expected class counts: Sketchy 94/10/21, TU-Berlin 200/20/30, QuickDraw 72/8/30. Train, val, and test class lists must be disjoint.

## Single training run (paper ASA-CLIP)

Sketchy-Ext, seed 42:

```bash
python main.py \
  --data_path /path/to/data \
  --output_dir ./outputs \
  --exp_name sketchy_asl_lambda0.7_db512_seed42 \
  --dataset sketchy \
  --seed 42 \
  --batch_size 32 \
  --num_workers 8 \
  --epochs 400 \
  --patience 10 \
  --lr 1e-5 \
  --weight_decay 0.05 \
  --validation_metric map@200 \
  --eval_metrics map@all map@200 p@200 p@100 \
  --use_semantic_alignment True \
  --use_text_adapter True \
  --text_bottleneck 512 \
  --lambda2 0.7 \
  --sketch_text_target adapted \
  --photo_loss_target raw \
  --eval_test_at_end True
```

TU-Berlin-Ext / QuickDraw-Ext: change `--dataset` and set `--validation_metric map@all`.

Each run writes:

- `training_run.log`
- `run_summary.log`
- `evaluation_metrics.csv`
- `val_classes.txt`
- `best_model_by_<metric>.pth`

## Three-seed main table

Main-table numbers in the revision use seeds **42 / 43 / 44**, frozen `lambda2=0.7` and `bottleneck=512`, sample standard deviation (`ddof=1`).

```bash
DATA_PATH=/path/to/data DEVICE=cuda:0 bash scripts/run_main_seeds.sh
python scripts/aggregate_seeds.py --roots ./revision_outputs/main_seeds
```

`scripts/run_main_seeds.sh` launches the three datasets sequentially on one GPU. Use the queue (below) if you want one job per GPU.

## Four-direction semantic-anchor ablation

Same protocol, seed 42:

| Name | `--sketch_text_target` | `--photo_loss_target` | `--use_text_adapter` |
|---|---|---|---|
| TE | raw | raw | False |
| TA | adapted | adapted | True |
| ASL (proposed) | adapted | raw | True |
| raw/adapted | raw | adapted | True |

```bash
DATA_PATH=/path/to/data SKIP_ASL=1 bash scripts/run_anchor_ablations.sh
```

`SKIP_ASL=1` reuses the main-table seed-42 ASL run.

## Sensitivity scans (Fig. lambda2 / bottleneck)

Paper main table stays at `lambda2=0.7`, `d_b=512`. Sensitivity jobs keep the other flag at the paper default:

- `lambda2` in `{0.1, 0.3, 0.5, 0.9}` with `--text_bottleneck 512`
- `bottleneck` in `{64, 128, 256, 1024}` with `--lambda2 0.7`

Do **not** rerun `0.7 / 512`; that point is the main-table seed-42 run.

Example (QuickDraw, `lambda2=0.1`):

```bash
python main.py \
  --data_path /path/to/data \
  --output_dir ./outputs/hyperparams \
  --exp_name quickdraw_l2_0.1_lambda0.1_db512_seed42 \
  --dataset quickdraw --seed 42 \
  --batch_size 32 --epochs 400 --patience 10 \
  --lr 1e-5 --weight_decay 0.05 \
  --validation_metric map@all \
  --eval_metrics map@all map@200 p@200 p@100 \
  --use_semantic_alignment True --use_text_adapter True \
  --text_bottleneck 512 --lambda2 0.1 \
  --sketch_text_target adapted --photo_loss_target raw \
  --eval_test_at_end True
```

Queue templates live in `scripts/queue/jobs/` (`*_l2_0.1_s42.conf`, `*_db64_s42.conf`, ...).

## Optional GPU queue

`scripts/queue/worker.sh` launches at most **two** `main.py` jobs across both GPUs (not two per GPU), and only if the chosen GPU has at least 7 GB free.

```bash
# edit scripts/queue/pending.txt, then:
screen -S asa_queue bash scripts/queue/worker.sh
```

Job files are `scripts/queue/jobs/<id>.conf`. See `scripts/queue/README.md`.

## Efficiency / bootstrap (no extra training)

```bash
python scripts/measure_efficiency.py --variant asa --text_bottleneck 512
python scripts/bootstrap_eval.py --help
```

FLOPs for ViT-B/16 are the handbook 17.6 GFLOPs / image; a sketch–photo pair is 35.2. The text adapter is unused at retrieval time.

## What this repository does not include

- Datasets and CLIP / training weights
- `revision_outputs/`, logs, and checkpoints
- GPU-occupy helpers used on a shared machine
- The frozen original submission tree

If you reproduce tables, do not mix numbers from a test-selected protocol with this independent-val protocol.
