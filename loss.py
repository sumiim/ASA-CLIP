import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Bidirectional InfoNCE loss used for visual and semantic alignment."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features_a, features_b):
        a = F.normalize(features_a.float(), p=2, dim=-1)
        b = F.normalize(features_b.float(), p=2, dim=-1)
        logits = torch.matmul(a, b.t()) / self.temperature
        targets = torch.arange(a.size(0), device=a.device)
        loss_ab = F.cross_entropy(logits, targets)
        loss_ba = F.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_ab + loss_ba)
