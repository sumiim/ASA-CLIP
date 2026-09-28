#!/usr/bin/env python3
"""Query-level paired bootstrap on official unseen test, from best-val ckpts.

Does not train. Default False for evaluate() is unchanged; this script is the
only caller of return_per_query=True.

n=3 training seeds is too small for Wilcoxon. Bootstrap resamples queries.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset import QuickDrawDataset, SketchyDataset  # noqa: E402
from evaluation import evaluate  # noqa: E402
from model import (  # noqa: E402
    ClipFitDualEncoderWithTextAdapter,
    ClipFitSingleEncoder,
    configure_clipfit_parameters,
    get_model,
)


def bootstrap_mean_ci(values, n_boot=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = values.size
    draws = rng.integers(0, n, size=(n_boot, n))
    means = values[draws].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "n_queries": int(n),
        "mean": float(values.mean()),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "n_boot": n_boot,
    }


def bootstrap_paired_diff(a, b, n_boot=10000, seed=0, alpha=0.05):
    if a.shape != b.shape:
        raise ValueError(f"paired arrays differ: {a.shape} vs {b.shape}")
    delta = a - b
    rng = np.random.default_rng(seed)
    n = delta.size
    draws = rng.integers(0, n, size=(n_boot, n))
    means = delta[draws].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "n_queries": int(n),
        "mean_diff": float(delta.mean()),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "frac_positive": float((means > 0).mean()),
        "n_boot": n_boot,
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def build_model(variant, bottleneck, device):
    if variant == "single":
        model = ClipFitSingleEncoder(model_name="ViT-B-16", pretrained="openai")
        configure_clipfit_parameters(model, train_sketch=True, train_photo=True)
    elif variant in {"dc", "dual_ln"}:
        model = ClipFitDualEncoderWithTextAdapter(
            model_name="ViT-B-16",
            pretrained="openai",
            use_text_adapter=False,
            text_bottleneck=bottleneck,
        )
        configure_clipfit_parameters(model, train_sketch=True, train_photo=True)
    elif variant == "asa":
        model = ClipFitDualEncoderWithTextAdapter(
            model_name="ViT-B-16",
            pretrained="openai",
            use_text_adapter=True,
            text_bottleneck=bottleneck,
        )
        configure_clipfit_parameters(model, train_sketch=True, train_photo=True)
    else:
        raise ValueError(variant)
    return model.to(device)


def load_ckpt(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {path} missing={len(missing)} unexpected={len(unexpected)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data_path", default="../data/")
    parser.add_argument("--dataset", default="sketchy", choices=["sketchy", "tuberlin", "quickdraw"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--text_bottleneck", type=int, default=512)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--out_json", required=True)
    parser.add_argument(
        "--job",
        action="append",
        nargs=2,
        metavar=("VARIANT", "CKPT"),
        required=True,
        help="Repeatable: --job asa path.pth --job dc path.pth",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    _, _, preprocess_val = get_model("ViT-B-16", "openai")
    data_dir_map = {
        "sketchy": "Sketchy_extended",
        "tuberlin": "TUBerlin",
        "quickdraw": "quickdraw_extended",
    }
    data_root = os.path.join(args.data_path, data_dir_map[args.dataset])
    dataset_class = SketchyDataset if args.dataset == "sketchy" else QuickDrawDataset
    test_dataset = dataset_class(
        root_dir=data_root,
        mode="test",
        dataset_name=args.dataset,
        split_mode="sketchy2",
        task_type="category",
        transform=preprocess_val,
        gzs=False,
        use_text_guidance=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    metric = "map@200" if args.dataset == "sketchy" else "map@all"
    eval_metrics = ["map@all", "map@200", "p@100", "p@200"]
    per_query = {}
    means = {}
    for variant, ckpt_path in args.job:
        print(f"=== {variant} {ckpt_path} ===")
        model = build_model(variant, args.text_bottleneck, device)
        load_ckpt(model, ckpt_path, device)
        model.eval()
        results = evaluate(
            model,
            test_loader,
            device,
            use_amp=True,
            eval_metrics=eval_metrics,
            return_per_query=True,
        )
        # evaluate() stores AP/P as fractions; convert to percent to match TEST logs.
        per_query[variant] = {k: np.asarray(v) * 100.0 for k, v in results["per_query"].items()}
        means[variant] = {k: float(results[k]) for k in eval_metrics}
        print("means", means[variant])
        del model
        torch.cuda.empty_cache()

    cis = {}
    for variant, arrays in per_query.items():
        cis[variant] = {m: bootstrap_mean_ci(arr, n_boot=args.n_boot, seed=0) for m, arr in arrays.items()}

    pairs = []
    names = list(per_query.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    metric: bootstrap_paired_diff(
                        per_query[a][metric],
                        per_query[b][metric],
                        n_boot=args.n_boot,
                        seed=0,
                    ),
                }
            )

    out = {
        "dataset": args.dataset,
        "primary_metric": metric,
        "means": means,
        "bootstrap_mean_ci": cis,
        "paired_diffs": pairs,
        "note": "Values in percent. Paired bootstrap resamples queries, not training seeds. n=3 Wilcoxon is not used.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    serial = json.loads(
        json.dumps(
            out,
            default=lambda o: o if not isinstance(o, np.generic) else o.item(),
        )
    )
    with open(args.out_json, "w", encoding="utf-8") as file:
        json.dump(serial, file, indent=2)
    print(f"Wrote {args.out_json}")
    print(json.dumps(serial["paired_diffs"], indent=2))


if __name__ == "__main__":
    main()
