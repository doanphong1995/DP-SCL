"""
Loss functions used by DP-SCL.

Reference sources:
  Supervised Contrastive Learning, Khosla et al., NeurIPS 2020:
  https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html

  SupContrast PyTorch reference implementation by HobbitLong:
  https://github.com/HobbitLong/SupContrast
"""

import torch
from torch import nn


class SupConLoss(nn.Module):
    """Supervised contrastive loss based on Khosla et al. and SupContrast."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        labels = labels.view(-1)
        if features.dim() == 3:
            batch_size, n_views, proj_dim = features.shape
            features = features.view(batch_size * n_views, proj_dim)
            labels = labels.repeat_interleave(n_views)

        batch_size = features.shape[0]
        if batch_size <= 1 or len(torch.unique(labels)) < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        similarity = torch.matmul(features, features.T) / self.temperature
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=features.device)
        mask = mask * logits_mask

        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_count = torch.clamp(mask.sum(dim=1), min=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / positive_count
        return -mean_log_prob.mean()
