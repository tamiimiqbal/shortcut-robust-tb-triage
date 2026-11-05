"""
Multimodal Fusion Modules (Section 4.2.3):
  1. `LateFusionModule`: Standard concatenation baseline
  2. `GatedCrossModalAttentionFusion`: Proposed bidirectional cross-modal attention + learned sigmoid gating
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object


if TORCH_AVAILABLE:
    class LateFusionModule(nn.Module):
        """
        Baseline late fusion: concatenates audio and clinical embeddings and projects
        to the shared representation space.
        """

        def __init__(self, audio_dim: int = 128, clinical_dim: int = 64, out_dim: int = 128, dropout: float = 0.25):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(audio_dim + clinical_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        def forward(self, audio_emb: torch.Tensor, clinical_emb: torch.Tensor, cough_tokens=None, cough_mask=None):
            concat = torch.cat([audio_emb, clinical_emb], dim=-1)
            fused = self.proj(concat)
            gate_mean = torch.full((audio_emb.size(0), 1), 0.5, device=audio_emb.device)
            return fused, gate_mean


    class GatedCrossModalAttentionFusion(nn.Module):
        """
        Proposed Gated Cross-Modal Attention Fusion (Section 4.2.3):
          1. Projects clinical features to query the sequence of patient cough tokens via Multi-Head Cross-Attention
             (allowing symptom context such as prolonged cough duration or hemoptysis to attend to specific cough bursts).
          2. Computes a learned sigmoid gate `g = sigma(W_g [a_ctx || c_proj])` that dynamically weights
             acoustic vs. clinical evidence per patient.
        """

        def __init__(
            self,
            audio_dim: int = 128,
            clinical_dim: int = 64,
            fusion_dim: int = 128,
            num_heads: int = 4,
            dropout: float = 0.25,
        ):
            super().__init__()
            self.audio_proj = nn.Linear(audio_dim, fusion_dim)
            self.clin_proj = nn.Sequential(
                nn.Linear(clinical_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
            )
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.gate_net = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.Sigmoid(),
            )
            self.out_norm = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Dropout(dropout),
            )

        def forward(
            self,
            audio_emb: torch.Tensor,
            clinical_emb: torch.Tensor,
            cough_tokens: torch.Tensor = None,
            cough_mask: torch.Tensor = None,
        ):
            a_vec = self.audio_proj(audio_emb)       # (B, D)
            c_vec = self.clin_proj(clinical_emb)     # (B, D)

            if cough_tokens is not None:
                # Clinical vector queries the K cough tokens via cross-attention
                query = c_vec.unsqueeze(1)           # (B, 1, D)
                key_padding_mask = (cough_mask <= 0.0) if cough_mask is not None else None
                attn_out, _ = self.cross_attn(
                    query=query,
                    key=cough_tokens,
                    value=cough_tokens,
                    key_padding_mask=key_padding_mask,
                )
                a_attended = a_vec + attn_out.squeeze(1)
            else:
                a_attended = a_vec

            gate = self.gate_net(torch.cat([a_attended, c_vec], dim=-1))  # (B, D) in (0, 1)
            fused = self.out_norm(gate * a_attended + (1.0 - gate) * c_vec)
            return fused, gate.mean(dim=-1, keepdim=True)
else:
    class LateFusionModule:
        pass

    class GatedCrossModalAttentionFusion:
        pass
