# Provenance

This public tree is copied from the **live** `HAPER_revision` working tree used for the ESWA revision experiments, not from the frozen original `HAPER/` directory and not from git commit `23dbc24` (that snapshot is the submission code *before* the independent-validation protocol).

Copied as-is from the running revision (uncommitted vs `23dbc24`):

- `dataset.py`, `evaluation.py`, `loss.py`, `main.py`, `trainer.py`, `utils.py`
- `scripts/run_main_seeds.sh`, `scripts/aggregate_seeds.py`, and the other public scripts listed in README

Public-only edits (do **not** exist in the live training tree):

- `model.py`: removed the machine-local CLIP path `/home/.../openai_clip_B_16/pytorch_model.bin`. Weights now come from `ASA_CLIP_WEIGHTS`, a file path in `--pretrained`, or the OpenCLIP `openai` tag.
- `scripts/queue/launch_job.sh` / `worker.sh`: default interpreter is `python`, not a conda path.
- Added `README.md`, `LICENSE`, `.gitignore`, `requirements.txt`, `pretrained/README.md`.
- Added missing main-table job templates `sketchy_asl_s42.conf`, `sketchy_asl_s43.conf`, `tuberlin_asl_s42.conf` (the live queue had already consumed those ids).

Excluded on purpose: datasets, checkpoints, `revision_outputs/`, logs, GPU-occupy scripts, reviewer Word/PDF files, and any nsec/key material.
