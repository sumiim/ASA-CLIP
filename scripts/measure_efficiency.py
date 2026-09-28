#!/usr/bin/env python3
"""Measure params / peak memory / latency for ASA-CLIP and two visual ablations.

Does not train. Optional --ckpt loads a best-val checkpoint for inference timing.
FLOPs for ViT-B/16 are the known 17.6 GFLOPs per image; dual-branch is 2x.
The text adapter is unused at retrieval time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model import (  # noqa: E402
    ClipFitDualEncoderWithTextAdapter,
    ClipFitSingleEncoder,
    configure_clipfit_parameters,
)


VIT_B16_GFLOPS = 17.6


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def peak_mb():
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_peak():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def latency_ms(fn, warmup=10, repeats=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / (len(times) - 1)
    return mean, var ** 0.5


def build_model(variant, args, device):
    if variant == "single":
        model = ClipFitSingleEncoder(model_name=args.model_name, pretrained=args.pretrained)
        configure_clipfit_parameters(model, train_sketch=True, train_photo=True)
    elif variant == "dual_ln":
        model = ClipFitDualEncoderWithTextAdapter(
            model_name=args.model_name,
            pretrained=args.pretrained,
            use_text_adapter=False,
            text_bottleneck=args.text_bottleneck,
        )
        configure_clipfit_parameters(model, train_sketch=True, train_photo=True)
    elif variant == "asa":
        model = ClipFitDualEncoderWithTextAdapter(
            model_name=args.model_name,
            pretrained=args.pretrained,
            use_text_adapter=True,
            text_bottleneck=args.text_bottleneck,
        )
        configure_clipfit_parameters(model, train_sketch=True, train_photo=True)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    # Train encodes a sketch+photo pair even for the shared single-branch encoder.
    # Retrieval encodes one image; the text adapter is unused at test time.
    gflops_train = VIT_B16_GFLOPS * 2
    gflops_infer = VIT_B16_GFLOPS
    return model.to(device), gflops_train, gflops_infer, gflops_infer


def measure_variant(variant, args, device):
    model, gflops_train, gflops_sketch, gflops_photo = build_model(variant, args, device)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[{variant}] loaded ckpt missing={len(missing)} unexpected={len(unexpected)}")

    total, trainable = count_params(model)
    dummy_s = torch.randn(args.batch_size, 3, 224, 224, device=device)
    dummy_p = torch.randn(args.batch_size, 3, 224, 224, device=device)
    dummy_t = torch.randn(args.batch_size, 512, device=device)

    reset_peak()
    model.train()
    sketch_feat, photo_feat, adapted = model(dummy_s, dummy_p, text_feat=dummy_t)
    loss = sketch_feat.mean() + photo_feat.mean()
    if adapted is not None:
        loss = loss + adapted.mean()
    loss.backward()
    torch.cuda.synchronize()
    train_peak = peak_mb()
    model.zero_grad(set_to_none=True)

    reset_peak()
    model.eval()
    with torch.no_grad():
        _ = model.encode_sketch(dummy_s)
        _ = model.encode_photo(dummy_p)
    torch.cuda.synchronize()
    infer_peak = peak_mb()

    one_s = dummy_s[:1]
    one_p = dummy_p[:1]
    sketch_mean, sketch_std = latency_ms(lambda: model.encode_sketch(one_s))
    photo_mean, photo_std = latency_ms(lambda: model.encode_photo(one_p))

    result = {
        "variant": variant,
        "total_params": int(total),
        "trainable_params": int(trainable),
        "train_peak_mb_batch32": round(train_peak, 2),
        "infer_peak_mb_batch32": round(infer_peak, 2),
        "gflops_train_forward_pair": gflops_train,
        "gflops_infer_sketch": gflops_sketch,
        "gflops_infer_photo": gflops_photo,
        "sketch_latency_ms_mean": round(sketch_mean, 3),
        "sketch_latency_ms_std": round(sketch_std, 3),
        "photo_latency_ms_mean": round(photo_mean, 3),
        "photo_latency_ms_std": round(photo_std, 3),
        "note": "Retrieval FLOPs/latency are visual-only; text adapter is unused at test time.",
    }
    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="Efficiency measurements for revision Table.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_name", type=str, default="ViT-B-16")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--text_bottleneck", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional best-val checkpoint.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["asa", "dual_ln", "single"],
        choices=["asa", "dual_ln", "single"],
    )
    parser.add_argument("--out_json", type=str, default="./revision_outputs/efficiency.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("measure_efficiency.py requires CUDA.")
    device = torch.device(args.device)

    results = []
    for variant in args.variants:
        print(f"Measuring {variant} ...")
        results.append(measure_variant(variant, args, device))
        print(json.dumps(results[-1], indent=2))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
