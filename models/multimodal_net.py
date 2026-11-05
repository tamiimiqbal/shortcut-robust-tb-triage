"""
Complete Two-Branch Audio + Clinical Network with Gated Cross-Modal Attention Fusion
and Site-Adversarial Gradient Reversal Head (Section 4.2).
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object

if TORCH_AVAILABLE and "PatientAudioEncoder" not in globals():
    try:
        from .audio_encoder import PatientAudioEncoder
        from .clinical_encoder import ClinicalMLPEncoder
        from .fusion import LateFusionModule, GatedCrossModalAttentionFusion
        from .adversarial import SiteAdversarialHead
    except ImportError:
        from models.audio_encoder import PatientAudioEncoder
        from models.clinical_encoder import ClinicalMLPEncoder
        from models.fusion import LateFusionModule, GatedCrossModalAttentionFusion
        from models.adversarial import SiteAdversarialHead


if TORCH_AVAILABLE:
    class ShortcutRobustMultimodalNet(nn.Module):
        """
        Unified Architecture supporting all proposal configurations & ablations:
          - `mode="proposed"`: Audio CNN-BiLSTM + Clinical MLP + Gated Cross-Modal Attention + Site-Adversarial GRL Head
          - `mode="late_fusion"`: Audio CNN-BiLSTM + Clinical MLP + Concatenation Late Fusion (no GRL)
          - `mode="audio_only"`: Audio CNN-BiLSTM only
          - `mode="clinical_only"`: Clinical MLP only
        """

        def __init__(
            self,
            clinical_in_dim: int = 28,
            audio_dim: int = 128,
            clinical_dim: int = 64,
            fusion_dim: int = 128,
            n_sites: int = 7,
            mode: str = "proposed",
            use_adversarial: bool = True,
            dropout: float = 0.25,
        ):
            super().__init__()
            self.mode = mode
            self.use_adversarial = use_adversarial

            self.audio_encoder = PatientAudioEncoder(
                cnn_dim=audio_dim,
                lstm_hidden=audio_dim // 2,
                out_dim=audio_dim,
                dropout=dropout,
            )
            self.clinical_encoder = ClinicalMLPEncoder(
                in_dim=clinical_in_dim,
                hidden_dim=clinical_dim,
                out_dim=clinical_dim,
                dropout=dropout,
            )

            if mode == "proposed":
                self.fusion = GatedCrossModalAttentionFusion(
                    audio_dim=audio_dim,
                    clinical_dim=clinical_dim,
                    fusion_dim=fusion_dim,
                    dropout=dropout,
                )
                shared_dim = fusion_dim
            elif mode == "late_fusion":
                self.fusion = LateFusionModule(
                    audio_dim=audio_dim,
                    clinical_dim=clinical_dim,
                    out_dim=fusion_dim,
                    dropout=dropout,
                )
                shared_dim = fusion_dim
            elif mode == "audio_only":
                self.fusion = None
                shared_dim = audio_dim
            elif mode == "clinical_only":
                self.fusion = None
                shared_dim = clinical_dim
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            self.tb_head = nn.Sequential(
                nn.Linear(shared_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            self.site_head = SiteAdversarialHead(
                in_dim=shared_dim,
                hidden_dim=64,
                n_sites=n_sites,
                dropout=dropout,
            )

        def forward(
            self,
            coughs: torch.Tensor,
            cough_mask: torch.Tensor,
            clinical: torch.Tensor,
            lambda_adv: float = 0.5,
        ):
            if self.mode == "audio_only":
                shared, _, attn_w = self.audio_encoder(coughs, cough_mask)
                gate_w = torch.ones((coughs.size(0), 1), device=coughs.device)
            elif self.mode == "clinical_only":
                shared = self.clinical_encoder(clinical)
                attn_w = cough_mask / cough_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                gate_w = torch.zeros((coughs.size(0), 1), device=coughs.device)
            else:
                a_emb, cough_tokens, attn_w = self.audio_encoder(coughs, cough_mask)
                c_emb = self.clinical_encoder(clinical)
                shared, gate_w = self.fusion(a_emb, c_emb, cough_tokens, cough_mask)

            tb_logits = self.tb_head(shared).squeeze(-1)
            eff_lambda = lambda_adv if self.use_adversarial else 0.0
            site_logits = self.site_head(shared, lambda_adv=eff_lambda)

            return {
                "tb_logits": tb_logits,
                "site_logits": site_logits,
                "shared_features": shared,
                "cough_attn_weights": attn_w,
                "modality_gate": gate_w,
            }
else:
    class ShortcutRobustMultimodalNet:
        pass
