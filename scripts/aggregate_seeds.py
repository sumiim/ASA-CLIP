#!/usr/bin/env python3
"""Aggregate final test_* rows from independent-val runs into mean +/- std.

Does not mix old test-selected numbers. Uses sample standard deviation (ddof=1).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict


METRICS = ["map@all", "map@200", "p@100", "p@200"]


def find_csvs(root):
    hits = []
    for dirpath, _, filenames in os.walk(root):
        if "evaluation_metrics.csv" in filenames:
            hits.append(os.path.join(dirpath, "evaluation_metrics.csv"))
    return sorted(hits)


def last_test_row(path):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    test_rows = [row for row in rows if row.get("split") == "test"]
    if not test_rows:
        return None
    return test_rows[-1]


def parse_float(value):
    if value is None or value == "":
        return None
    return float(value)


def infer_group(path, run_dir_name):
    name = run_dir_name.lower()
    dataset = "unknown"
    for candidate in ("sketchy", "tuberlin", "quickdraw"):
        if candidate in name:
            dataset = candidate
            break
    seed = None
    if "seed" in name:
        token = name.split("seed", 1)[1]
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            seed = int(digits)
    variant = "unknown"
    padded = f"_{name}_"
    if "_raw_adapted_" in padded or name.startswith("raw_adapted"):
        variant = "raw_adapted"
    elif "_asl_" in padded or name.startswith("asl"):
        variant = "asl"
    elif "_te_" in padded or name.startswith("te_"):
        variant = "te"
    elif "_ta_" in padded or name.startswith("ta_"):
        variant = "ta"
    return dataset, variant, seed, path


def mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def main():
    parser = argparse.ArgumentParser(description="Aggregate test metrics from revision runs.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["./revision_outputs/main_seeds", "./revision_outputs/anchor_ablations"],
        help="Run directories to scan.",
    )
    parser.add_argument("--out_csv", type=str, default="./revision_outputs/aggregated_test_metrics.csv")
    args = parser.parse_args()

    groups = defaultdict(list)
    missing = []
    for root in args.roots:
        if not os.path.exists(root):
            print(f"Skip missing root: {root}")
            continue
        for csv_path in find_csvs(root):
            run_dir = os.path.basename(os.path.dirname(csv_path))
            dataset, variant, seed, path = infer_group(csv_path, run_dir)
            row = last_test_row(csv_path)
            if row is None:
                missing.append(csv_path)
                continue
            record = {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "run_dir": run_dir,
                "path": path,
            }
            for metric in METRICS:
                record[metric] = parse_float(row.get(f"test_{metric}"))
            groups[(dataset, variant)].append(record)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    print("Per-run test metrics")
    print("-" * 80)
    for (dataset, variant), records in sorted(groups.items()):
        print(f"[{dataset} | {variant}] n={len(records)}")
        for rec in sorted(records, key=lambda r: (r["seed"] is None, r["seed"] or 0, r["run_dir"])):
            metric_str = " ".join(
                f"{m}={rec[m]:.4f}" if rec[m] is not None else f"{m}=NA" for m in METRICS
            )
            print(f"  seed={rec['seed']} {metric_str}  {rec['run_dir']}")

    print()
    print("Mean +/- sample std (ddof=1); do not mix with old test-selected numbers")
    print("-" * 80)
    out_rows = []
    for (dataset, variant), records in sorted(groups.items()):
        out = {"dataset": dataset, "variant": variant, "n": len(records)}
        for metric in METRICS:
            values = [rec[metric] for rec in records if rec[metric] is not None]
            mean, std = mean_std(values)
            if mean is None:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_std"] = ""
                print(f"{dataset:10s} {variant:12s} {metric:8s}  n={len(values)}  NA")
            else:
                out[f"{metric}_mean"] = f"{mean:.4f}"
                out[f"{metric}_std"] = f"{std:.4f}"
                print(
                    f"{dataset:10s} {variant:12s} {metric:8s}  "
                    f"{mean:.2f} +/- {std:.2f}  (n={len(values)})"
                )
        out_rows.append(out)

    fieldnames = ["dataset", "variant", "n"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with open(args.out_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {args.out_csv}")
    if missing:
        print("Runs without a test row:")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()
