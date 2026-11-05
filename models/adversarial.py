"""
Gradient Reversal Layer (GRL) and Site-Adversarial Classifier Head (Section 4.2.5).

Reference:
  Ganin, Y., et al. (2016). Domain-Adversarial Training of Neural Networks. JMLR, 17(59).
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.autograd import Function
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object
    Function = object


def compute_dann_lambda(progress: float, max_lambda: float = 0.65, gamma: float = 10.0) -> float:
    """
    Smooth DANN adaptation schedule from 0 to max_lambda as training progress p goes from 0 -> 1.
    """
    p = float(np.clip(progress, 0.0, 1.0))
    val = (2.0 / (1.0 + np.exp(-gamma * p))) - 1.0
    return float(max_lambda * val)


if TORCH_AVAILABLE:
    class _GradientReversalFunction(Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, lambda_adv: float) -> torch.Tensor:
            ctx.lambda_adv = float(lambda_adv)
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            return grad_output.neg() * ctx.lambda_adv, None


    class GradientReversalLayer(nn.Module):
        """
        Identity in forward pass; multiplies gradient by -lambda_adv in backward pass.
        Pushes shared features to become invariant to recording site/country.
        """

        def __init__(self, lambda_adv: float = 0.5):
            super().__init__()
            self.lambda_adv = lambda_adv

        def set_lambda(self, lambda_adv: float) -> None:
            self.lambda_adv = float(lambda_adv)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.lambda_adv <= 0.0:
                return x
            return _GradientReversalFunction.apply(x, self.lambda_adv)


    class SiteAdversarialHead(nn.Module):
        """
        Predicts recording country/site (7 CODA countries) from shared representation
        through a Gradient Reversal Layer.
        """

        def __init__(self, in_dim: int = 128, hidden_dim: int = 64, n_sites: int = 7, dropout: float = 0.25):
            super().__init__()
            self.grl = GradientReversalLayer(lambda_adv=0.5)
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_sites),
            )

        def forward(self, shared_features: torch.Tensor, lambda_adv: float = 0.5) -> torch.Tensor:
            self.grl.set_lambda(lambda_adv)
            reversed_feat = self.grl(shared_features)
            return self.classifier(reversed_feat)
else:
    class GradientReversalLayer:
        def __init__(self, lambda_adv: float = 0.5):
            self.lambda_adv = lambda_adv

    class SiteAdversarialHead:
        def __init__(self, in_dim: int = 128, hidden_dim: int = 64, n_sites: int = 7, dropout: float = 0.25):
            self.in_dim = in_dim
            self.n_sites = n_sites
