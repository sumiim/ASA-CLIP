import argparse
import os
from datetime import datetime

import open_clip
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import QuickDrawDataset, SketchyDataset
from model import (
    ClipFitDualEncoderWithTextAdapter,
    ClipFitSingleEncoder,
    configure_clipfit_parameters,
    get_model,
)
from trainer import Trainer
from utils import get_logger, set_seed


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in {"true", "1", "yes", "y"}


def build_text_bank(args, train_dataset, device, logger):
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    sorted_classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    prompts = [f"a photo of a {name.replace('_', ' ')}" for name in sorted_classes]

    clip_model, _, _ = get_model(args.model_name, args.pretrained, logger)
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer(args.model_name)
    text_tokens = tokenizer(prompts)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()
        text_features = F.normalize(text_features, p=2, dim=-1)

    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text_features.to(device)


def build_optimizer(model, args):
    backbone_decay, backbone_no_decay = [], []
    adapter_decay, adapter_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = name.endswith("bias") or "norm" in name.lower() or "ln" in name.lower()
        if "text_adapter" in name:
            (adapter_no_decay if is_no_decay else adapter_decay).append(param)
        else:
            (backbone_no_decay if is_no_decay else backbone_decay).append(param)

    param_groups = [
        {"params": backbone_decay, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": backbone_no_decay, "lr": args.lr, "weight_decay": 0.0},
        {
            "params": adapter_decay,
            "lr": args.lr * args.text_adapter_lr_ratio,
            "weight_decay": args.weight_decay,
        },
        {
            "params": adapter_no_decay,
            "lr": args.lr * args.text_adapter_lr_ratio,
            "weight_decay": 0.0,
        },
    ]
    param_groups = [group for group in param_groups if len(group["params"]) > 0]
    return optim.AdamW(param_groups)


def main(args):
    if not 0.0 <= args.lambda2 <= 1.0:
        raise ValueError("--lambda2 must be in [0, 1].")
    if args.use_semantic_alignment and args.lambda2 == 0.0:
        raise ValueError("--use_semantic_alignment true requires --lambda2 > 0.")
    if not args.use_semantic_alignment:
        args.lambda2 = 0.0

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_str = f"{args.dataset}_category"
    branch_str = "single" if args.use_single_branch else "dual"
    sem_str = (
        f"sem_{args.sketch_text_target}_{args.photo_loss_target}_l2{args.lambda2}"
        if args.use_semantic_alignment
        else "cf_only"
    )
    run_name = f"{args.exp_name}_{timestamp}" if args.exp_name else f"{task_str}_{branch_str}_{sem_str}_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    logger = get_logger(output_dir=run_dir, log_name="training_run.log")
    logger.info(f"Run directory: {run_dir}")
    logger.info("Hyperparameters")
    logger.info(f"dataset={args.dataset}, model={args.model_name}, pretrained={args.pretrained}")
    logger.info(f"batch_size={args.batch_size}, epochs={args.epochs}, patience={args.patience}")
    logger.info(f"lr={args.lr}, weight_decay={args.weight_decay}")
    logger.info(f"temperature_v={args.temperature_v}, temperature_sem={args.temperature_sem}")
    logger.info(f"lambda2={args.lambda2}")
    logger.info(
        f"single_branch={args.use_single_branch}, train_sketch={args.train_sketch}, "
        f"train_photo={args.train_photo}"
    )
    logger.info(
        f"semantic_alignment={args.use_semantic_alignment}, text_adapter={args.use_text_adapter}, "
        f"text_bottleneck={args.text_bottleneck}, text_adapter_lr_ratio={args.text_adapter_lr_ratio}"
    )
    logger.info(
        f"sketch_text_target={args.sketch_text_target}, photo_loss_target={args.photo_loss_target}"
    )
    logger.info(f"validation_metric={args.validation_metric}, eval_metrics={args.eval_metrics}")

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    temp_model, preprocess_train, preprocess_val = get_model(
        args.model_name, args.pretrained, logger=logger
    )
    del temp_model

    data_dir_map = {
        "sketchy": "Sketchy_extended",
        "tuberlin": "TUBerlin",
        "quickdraw": "quickdraw_extended",
    }
    data_root = os.path.join(args.data_path, data_dir_map.get(args.dataset, args.dataset))
    logger.info(f"Data root: {data_root}")
    dataset_class = SketchyDataset if args.dataset == "sketchy" else QuickDrawDataset
    train_dataset = dataset_class(
        root_dir=data_root,
        mode="train",
        dataset_name=args.dataset,
        split_mode="sketchy2",
        task_type="category",
        transform_sketch=preprocess_train,
        transform_photo=preprocess_train,
        logger=logger,
        use_text_guidance=False,
    )
    test_dataset = dataset_class(
        root_dir=data_root,
        mode="test",
        dataset_name=args.dataset,
        split_mode="sketchy2",
        task_type="category",
        transform=preprocess_val,
        logger=logger,
        gzs=False,
        use_text_guidance=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    if args.use_single_branch:
        model = ClipFitSingleEncoder(
            model_name=args.model_name, pretrained=args.pretrained, logger=logger
        )
    else:
        model = ClipFitDualEncoderWithTextAdapter(
            model_name=args.model_name,
            pretrained=args.pretrained,
            use_text_adapter=args.use_text_adapter,
            text_bottleneck=args.text_bottleneck,
            logger=logger,
        )
    model.to(device)

    text_bank = None
    if args.use_semantic_alignment:
        text_bank = build_text_bank(args, train_dataset, device, logger)

    trainable_params = configure_clipfit_parameters(
        model,
        logger=logger,
        train_sketch=args.train_sketch,
        train_photo=args.train_photo,
    )
    optimizer = build_optimizer(model, args)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        run_dir=run_dir,
        patience=args.patience,
        logger=logger,
        trainable_params=trainable_params,
        validation_metric=args.validation_metric,
        eval_metrics=args.eval_metrics,
        text_bank=text_bank,
        temperature_v=args.temperature_v,
        temperature_sem=args.temperature_sem,
        use_semantic_alignment=args.use_semantic_alignment,
        use_text_adapter=args.use_text_adapter,
        lambda2=args.lambda2,
        sketch_text_target=args.sketch_text_target,
        photo_loss_target=args.photo_loss_target,
    )
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper-aligned ClipFit ZS-SBIR training")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="sketchy", choices=["sketchy", "tuberlin", "quickdraw"])
    parser.add_argument("--model_name", type=str, default="ViT-B-16")
    parser.add_argument("--pretrained", type=str, default="openai")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--validation_metric", type=str, default="map@all")
    parser.add_argument("--eval_metrics", nargs="+", default=["map@all", "map@200", "p@100", "p@200"])

    parser.add_argument("--temperature_v", type=float, default=0.07)
    parser.add_argument("--temperature_sem", type=float, default=0.07)
    parser.add_argument(
        "--lambda2",
        type=float,
        default=0.5,
        help="Semantic loss weight. The visual weight lambda1 is computed as 1 - lambda2.",
    )

    parser.add_argument("--use_single_branch", type=str2bool, default=False)
    parser.add_argument("--train_sketch", type=str2bool, default=True)
    parser.add_argument("--train_photo", type=str2bool, default=True)

    parser.add_argument("--use_semantic_alignment", type=str2bool, default=False)
    parser.add_argument("--use_text_adapter", type=str2bool, default=False)
    parser.add_argument("--text_bottleneck", type=int, default=256)
    parser.add_argument("--text_adapter_lr_ratio", type=float, default=3.0)
    parser.add_argument(
        "--sketch_text_target",
        type=str,
        default="adapted",
        choices=["raw", "adapted"],
        help="Sketch semantic target for TE/TA/ASL ablations.",
    )
    parser.add_argument(
        "--photo_loss_target",
        type=str,
        default="none",
        choices=["none", "raw", "adapted"],
        help="Photo semantic target. Use raw for ASL and adapted for symmetric TA.",
    )

    main(parser.parse_args())
