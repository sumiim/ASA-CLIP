import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from evaluation import evaluate
from loss import ContrastiveLoss


class Trainer:
    """Trainer for the paper model and its ablations.

    Model selection uses an independent seen-class val loader. The official
    unseen test loader is evaluated once at the end from the best val checkpoint.
    """

    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        device,
        epochs,
        run_dir,
        patience,
        logger,
        trainable_params,
        validation_metric="map@all",
        eval_metrics=None,
        text_bank=None,
        temperature_v=0.07,
        temperature_sem=0.07,
        use_semantic_alignment=False,
        use_text_adapter=False,
        lambda2=0.5,
        sketch_text_target="adapted",
        photo_loss_target="none",
        scheduler=None,
        eval_test_at_end=True,
        summary=None,
    ):
        if not 0.0 <= lambda2 <= 1.0:
            raise ValueError("--lambda2 must be in [0, 1]. lambda1 is computed as 1 - lambda2.")
        if sketch_text_target not in {"raw", "adapted"}:
            raise ValueError("sketch_text_target must be 'raw' or 'adapted'.")
        if photo_loss_target not in {"none", "raw", "adapted"}:
            raise ValueError("photo_loss_target must be 'none', 'raw', or 'adapted'.")
        if val_loader is None:
            raise ValueError("val_loader is required; do not use the test set for model selection.")

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.run_dir = run_dir
        self.patience = patience
        self.logger = logger
        self.trainable_params = trainable_params
        self.validation_metric = validation_metric
        self.eval_metrics = eval_metrics if eval_metrics else ["map@all", "map@200", "p@100", "p@200"]
        self.text_bank = text_bank
        self.use_semantic_alignment = use_semantic_alignment
        self.use_text_adapter = use_text_adapter
        self.lambda2 = float(lambda2)
        self.sketch_text_target = sketch_text_target
        self.photo_loss_target = photo_loss_target
        self.scheduler = scheduler
        self.eval_test_at_end = eval_test_at_end
        self.summary = summary

        self.clipat_triplet = getattr(model, "sk_prompt", None) is not None
        self.marl_method = hasattr(model, "marl_loss")
        if self.clipat_triplet:
            self.visual_loss_fn = nn.TripletMarginWithDistanceLoss(
                distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
                margin=0.2,
            ).to(self.device)
        else:
            self.visual_loss_fn = ContrastiveLoss(temperature=temperature_v).to(self.device)
        self.semantic_loss_fn = ContrastiveLoss(temperature=temperature_sem).to(self.device)

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        self.epochs_no_improve = 0
        self.best_metric = float("-inf")
        self.best_epoch_metrics = {}
        self.best_epoch = 0
        self.test_metrics = {}

        self.metrics_csv_path = os.path.join(self.run_dir, "evaluation_metrics.csv")
        self._initialize_csv()

    def _log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def _initialize_csv(self):
        self.csv_header = [
            "epoch",
            "train_loss",
            "train_visual_loss",
            "train_semantic_loss",
            "lambda2",
            "validation_metric_value",
            "split",
        ]
        for metric in self.eval_metrics:
            self.csv_header.append(f"val_{metric}")
        for metric in self.eval_metrics:
            self.csv_header.append(f"test_{metric}")
        with open(self.metrics_csv_path, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(self.csv_header)

    def _log_metrics_to_csv(self, metrics_dict):
        row = [metrics_dict.get(col, "") for col in self.csv_header]
        with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(row)

    def _semantic_targets(self, target_text, adapted_text):
        sketch_target = target_text if self.sketch_text_target == "raw" else adapted_text
        if self.photo_loss_target == "raw":
            photo_target = target_text
        elif self.photo_loss_target == "adapted":
            photo_target = adapted_text
        else:
            photo_target = None
        return sketch_target, photo_target

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_visual_loss = 0.0
        total_semantic_loss = 0.0
        use_amp = self.device.type == "cuda"

        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}", leave=False)
        for batch in pbar:
            sketches = batch[0].to(self.device, non_blocking=True)
            photos = batch[1].to(self.device, non_blocking=True)
            labels = batch[2].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                target_text = None
                if self.use_semantic_alignment:
                    if self.text_bank is None:
                        raise RuntimeError("Semantic alignment is enabled, but text_bank is None.")
                    target_text = self.text_bank[labels]

                if self.marl_method:
                    if self.text_bank is None:
                        raise RuntimeError("MARL requires text_bank.")
                    target_text = self.text_bank[labels]
                    visual_loss, _aux = self.model.marl_loss(
                        sketches, photos, target_text, self.semantic_loss_fn
                    )
                    adapted_text = None
                    sketch_feat = photo_feat = None
                else:
                    sketch_feat, photo_feat, adapted_text = self.model(
                        sketches, photos, text_feat=target_text
                    )

                if self.marl_method:
                    pass
                elif self.clipat_triplet:
                    # CLIP-AT triplet: positive is the paired photo; negative is the
                    # next in-batch photo (cyclic). Same paired sampling as our loader.
                    neg_feat = torch.roll(photo_feat, shifts=1, dims=0)
                    visual_loss = self.visual_loss_fn(sketch_feat, photo_feat, neg_feat)
                else:
                    visual_loss = self.visual_loss_fn(sketch_feat, photo_feat)
                semantic_loss = torch.tensor(0.0, device=self.device)
                active_lambda2 = 0.0

                if self.use_semantic_alignment:
                    sketch_target, photo_target = self._semantic_targets(target_text, adapted_text)
                    l_sem_sketch = self.semantic_loss_fn(sketch_feat, sketch_target)
                    l_sem_photo = (
                        self.semantic_loss_fn(photo_feat, photo_target)
                        if photo_target is not None
                        else torch.tensor(0.0, device=self.device)
                    )
                    semantic_loss = l_sem_sketch + l_sem_photo
                    active_lambda2 = self.lambda2

                loss = (1.0 - active_lambda2) * visual_loss + active_lambda2 * semantic_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_visual_loss += visual_loss.item()
            total_semantic_loss += semantic_loss.item()
            pbar.set_postfix(
                L=f"{loss.item():.3f}",
                V=f"{visual_loss.item():.3f}",
                S=f"{semantic_loss.item():.3f}",
            )

        if self.scheduler is not None:
            self.scheduler.step()

        num_batches = len(self.train_loader)
        return (
            total_loss / num_batches,
            total_visual_loss / num_batches,
            total_semantic_loss / num_batches,
        )

    def checkpoint_path(self):
        return os.path.join(self.run_dir, f"best_model_by_{self.validation_metric}.pth")

    def validate(self):
        """Evaluate on the independent seen-class validation split only."""
        self.model.eval()
        metrics = evaluate(
            self.model,
            self.val_loader,
            self.device,
            self.logger,
            use_amp=self.device.type == "cuda",
            eval_metrics=self.eval_metrics,
        )
        return metrics.get(self.validation_metric, 0.0), metrics

    def evaluate_test(self):
        """Evaluate the official unseen test split once."""
        self.model.eval()
        metrics = evaluate(
            self.model,
            self.test_loader,
            self.device,
            self.logger,
            use_amp=self.device.type == "cuda",
            eval_metrics=self.eval_metrics,
        )
        return metrics

    def save_checkpoint(self, epoch, metric_value):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "metric": metric_value,
                "lambda2": self.lambda2,
            },
            self.checkpoint_path(),
        )

    def load_best_checkpoint(self):
        path = self.checkpoint_path()
        if not os.path.exists(path):
            self._log(f"No best checkpoint found at {path}; evaluating the current weights.")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self._log(
            f"Loaded best checkpoint from epoch {ckpt.get('epoch', '?')} "
            f"(val {self.validation_metric}={ckpt.get('metric', float('nan')):.4f})."
        )

    def run(self):
        if getattr(self, "marl_method", False):
            mode_desc = "marl, shared-ViT, LN+mode-embedding"
        elif getattr(self, "clipat_triplet", False):
            mode_desc = "clipat, shared-ViT, LN+visual-prompts, triplet"
        elif hasattr(self.model, "shared_model"):
            mode_desc = "single-branch"
        else:
            mode_desc = "dual-branch"
        if self.use_semantic_alignment:
            mode_desc += f", semantic={self.sketch_text_target}/{self.photo_loss_target}"
        self._log(f"Start training ({mode_desc}, lambda2={self.lambda2:.3f}).")
        self._log(
            "Model selection uses independent seen-class val; official test is evaluated once at the end."
        )
        if self.summary:
            self.summary.write("[train]")
            self.summary.kv(mode=mode_desc, lambda2=self.lambda2, max_epochs=self.epochs)

        for epoch in range(1, self.epochs + 1):
            train_loss, visual_loss, semantic_loss = self.train_one_epoch(epoch)

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_visual_loss": visual_loss,
                "train_semantic_loss": semantic_loss,
                "lambda2": self.lambda2,
                "split": "val",
            }

            val_score, val_metrics = self.validate()
            metrics["validation_metric_value"] = val_score
            disp = []
            if val_metrics:
                disp = [f"{k}={v:.2f}%" for k, v in val_metrics.items()]
                for key, value in val_metrics.items():
                    metrics[f"val_{key}"] = value

            result_parts = [
                f"Epoch {epoch}/{self.epochs}",
                f"Loss={train_loss:.4f}",
                f"Visual={visual_loss:.4f}",
                f"Semantic={semantic_loss:.4f}",
                "val " + ", ".join(disp) if disp else "val (empty)",
            ]
            self._log(" | ".join(result_parts))

            self._log_metrics_to_csv(metrics)
            if self.summary:
                improved = val_score > self.best_metric
                self.summary.kv(
                    epoch=epoch,
                    train_loss=train_loss,
                    visual=visual_loss,
                    semantic=semantic_loss,
                    best=int(improved),
                )
                self.summary.metrics("val", val_metrics)

            if val_score > self.best_metric:
                self.best_metric = val_score
                self.best_epoch_metrics = val_metrics.copy()
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, val_score)
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                self._log("Early stopping.")
                if self.summary:
                    self.summary.write(f"early_stop=1 epoch={epoch} patience={self.patience}")
                break

        if self.best_epoch_metrics:
            details = [f"{k}={v:.2f}%" for k, v in self.best_epoch_metrics.items()]
            self._log(
                f"Best val {self.validation_metric}={self.best_metric:.4f} "
                f"(epoch {self.best_epoch}) | {', '.join(details)}"
            )
            if self.summary:
                self.summary.write("[best_val]")
                self.summary.kv(epoch=self.best_epoch, metric=self.validation_metric, value=self.best_metric)
                self.summary.metrics("best_val", self.best_epoch_metrics)

        if self.eval_test_at_end:
            self.load_best_checkpoint()
            self.test_metrics = self.evaluate_test()
            test_disp = [f"{k}={v:.2f}%" for k, v in self.test_metrics.items()]
            self._log("TEST (once, best val ckpt) | " + ", ".join(test_disp))
            test_row = {
                "epoch": self.best_epoch,
                "lambda2": self.lambda2,
                "split": "test",
                "validation_metric_value": self.test_metrics.get(self.validation_metric, ""),
            }
            for key, value in self.best_epoch_metrics.items():
                test_row[f"val_{key}"] = value
            for key, value in self.test_metrics.items():
                test_row[f"test_{key}"] = value
            self._log_metrics_to_csv(test_row)
            if self.summary:
                self.summary.write("[test]")
                self.summary.kv(epoch=self.best_epoch, ckpt=self.checkpoint_path())
                self.summary.metrics("test", self.test_metrics)
