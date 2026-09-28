import csv
import os

import torch
from tqdm import tqdm

from evaluation import evaluate
from loss import ContrastiveLoss


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_loader,
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
    ):
        if not 0.0 <= lambda2 <= 1.0:
            raise ValueError("--lambda2 must be in [0, 1]. lambda1 is computed as 1 - lambda2.")
        if sketch_text_target not in {"raw", "adapted"}:
            raise ValueError("sketch_text_target must be 'raw' or 'adapted'.")
        if photo_loss_target not in {"none", "raw", "adapted"}:
            raise ValueError("photo_loss_target must be 'none', 'raw', or 'adapted'.")

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
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

        self.visual_loss_fn = ContrastiveLoss(temperature=temperature_v).to(self.device)
        self.semantic_loss_fn = ContrastiveLoss(temperature=temperature_sem).to(self.device)

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        self.epochs_no_improve = 0
        self.best_metric = float("-inf")
        self.best_epoch_metrics = {}

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
        ]
        for metric in self.eval_metrics:
            self.csv_header.append(f"val_{metric}")
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

                sketch_feat, photo_feat, adapted_text = self.model(
                    sketches, photos, text_feat=target_text
                )

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

    def validate(self):
        self.model.eval()
        metrics = evaluate(
            self.model,
            self.test_loader,
            self.device,
            self.logger,
            use_amp=self.device.type == "cuda",
            eval_metrics=self.eval_metrics,
        )
        return metrics.get(self.validation_metric, 0.0), metrics

    def save_checkpoint(self, epoch, metric_value):
        path = os.path.join(self.run_dir, f"best_model_by_{self.validation_metric}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "metric": metric_value,
                "lambda2": self.lambda2,
            },
            path,
        )

    def run(self):
        mode_desc = "single-branch" if hasattr(self.model, "shared_model") else "dual-branch"
        if self.use_semantic_alignment:
            mode_desc += f", semantic={self.sketch_text_target}/{self.photo_loss_target}"
        self._log(f"Start training ({mode_desc}, lambda2={self.lambda2:.3f}).")

        for epoch in range(1, self.epochs + 1):
            train_loss, visual_loss, semantic_loss = self.train_one_epoch(epoch)

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_visual_loss": visual_loss,
                "train_semantic_loss": semantic_loss,
                "lambda2": self.lambda2,
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
            ]
            if disp:
                result_parts.append(", ".join(disp))
            self._log(" | ".join(result_parts))

            self._log_metrics_to_csv(metrics)

            if val_score > self.best_metric:
                self.best_metric = val_score
                self.best_epoch_metrics = val_metrics.copy()
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, val_score)
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                self._log("Early stopping.")
                break

        if self.best_epoch_metrics:
            details = [f"{k}={v:.2f}%" for k, v in self.best_epoch_metrics.items()]
            self._log(f"Best {self.validation_metric}={self.best_metric:.4f} | {', '.join(details)}")
