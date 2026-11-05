"""
Imbalance-Aware Losses for TB Triage (Section 4.2.4):
  - FocalLossWithLogits (handles class imbalance + site-balanced sample weights)
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object


if TORCH_AVAILABLE:
    class FocalLossWithLogits(nn.Module):
        """
        Binary Focal Loss supporting per-patient site-balanced sample weights.
        """

        def __init__(self, alpha: float = 0.70, gamma: float = 2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            sample_weights: torch.Tensor = None,
        ) -> torch.Tensor:
            logits = logits.reshape(-1)
            targets = targets.reshape(-1).float()
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_term = alpha_t * ((1.0 - p_t).clamp(min=1e-6) ** self.gamma) * bce
            if sample_weights is not None:
                focal_term = focal_term * sample_weights.reshape(-1)
            return focal_term.mean()
else:
    class FocalLossWithLogits:
        pass
