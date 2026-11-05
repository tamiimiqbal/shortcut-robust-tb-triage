"""
Clinical Questionnaire MLP Encoder (Section 4.2.2).
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object


if TORCH_AVAILABLE:
    class ClinicalMLPEncoder(nn.Module):
        """
        Encodes preprocessed patient clinical & demographic questionnaire vector
        (numeric z-scores + one-hot symptoms + missingness indicators) into a
        dense clinical embedding `h_clin`.
        """

        def __init__(self, in_dim: int = 28, hidden_dim: int = 64, out_dim: int = 64, dropout: float = 0.25):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
else:
    class ClinicalMLPEncoder:
        pass
