# ASA-CLIP

Code for **ASA-CLIP: Asymmetric Semantic Anchoring for Zero-Shot Sketch-Based Image Retrieval (ZS-SBIR)**.

## Main settings

- Backbone: OpenCLIP ViT-B/16 (OpenAI weights)
- Trainable parts: LayerNorm parameters in the sketch and photo branches, plus the text adapter
- Batch size: `32`
- Epochs / patience: `400 / 10`
- Optimizer: AdamW, learning rate `1e-5`, weight decay `0.05`
- ASL weight: `lambda2=0.7`
- Text bottleneck: `512`
- Seeds: `42, 43, 44`
- Model selection: independent seen-class validation split; the official unseen test split is evaluated after selecting the best validation checkpoint

## Directory structure

```text
ASA-CLIP/
├── dataset.py              # Dataset loaders and fixed class splits
├── model.py                # ASA-CLIP and baseline model definitions
├── loss.py                 # Contrastive losses
├── trainer.py              # Training and validation loop
├── evaluation.py           # Retrieval evaluation
├── main.py                 # Training entry point
├── utils.py                # Logging and utilities
├── scripts/
│   ├── run_main_seeds.sh
│   ├── run_anchor_ablations.sh
│   ├── aggregate_seeds.py
│   ├── check_val_split.py
│   ├── measure_efficiency.py
│   └── bootstrap_eval.py
├── pretrained/             # Local checkpoint instructions only
├── requirements.txt
├── environment.yml
└── LICENSE
```

## Installation

```bash
conda create -n asa-clip python=3.10 -y
conda activate asa-clip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

The code uses an OpenCLIP ViT-B/16 OpenAI checkpoint. It can download the
checkpoint automatically, or load a local file through `ASA_CLIP_WEIGHTS` or
`--pretrained`. Checkpoints are not included in this repository.

## Datasets

The datasets are not included. Obtain them from their official or project
releases, following their individual licenses:

- [Sketchy](https://sketchy.eye.gatech.edu/) and the Sketchy-Ext photo extension
- TU-Berlin sketches and the TU-Berlin-Ext photo extension used in ZS-SBIR
- QuickDraw-Ext from the corresponding ZS-SBIR release

Place them under one data directory with this layout:

```text
<data_path>/
├── Sketchy_extended/
│   ├── sketch/tx_000000000000/<class>/
│   ├── photo/tx_000000000000/<class>/
│   └── extended_photo/<class>/
├── TUBerlin/
│   ├── png_ready/<class>/
│   └── ImageResized_ready/<class>/
└── quickdraw_extended/
    ├── sketches/<class>/
    └── photos/<class>/
```

Check the fixed train/validation/test class split with:

```bash
python scripts/check_val_split.py /path/to/data
```

See the scripts directory for the main experiments, ablations, sensitivity
scans, and result aggregation.
