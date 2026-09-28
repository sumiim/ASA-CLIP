# Optional training queue

Scheduler: `bash scripts/queue/worker.sh` (typically inside `screen`).

Rules:

- At most **2** `python -u main.py` jobs across **both** GPUs (not 2 per GPU)
- Launch only if the chosen GPU has at least **7 GB** free
- Prefer the GPU with fewer of our jobs and more free memory
- An empty queue idles; it does not occupy a GPU

```text
scripts/queue/
  jobs/<id>.conf     hyperparameters
  pending.txt        remaining job ids (top to bottom)
  pop_job.py         atomic pop of the next id
  worker.sh          poll GPUs and launch
  launch_job.sh      source conf, set CUDA_VISIBLE_DEVICES, run main.py
```

`PYTHON` and `DATA_PATH` may be overridden in the environment. Defaults are `python` and `../data/`.

To enqueue, put an id on a non-comment line of `pending.txt`. The matching `jobs/<id>.conf` must exist.
