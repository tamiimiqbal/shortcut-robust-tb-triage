"""
Audio Branch: 2D Log-Mel Spectrogram CNN + BiLSTM + Multi-Cough Attention Pooling (Section 4.2.1).
Also includes a Pretrained Audio Encoder comparison module.
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
    class CoughCNN2D(nn.Module):
        """
        Extracts a compact cough-level acoustic vector from a single 0.5s (1, 64, 32)
        Log-Mel Spectrogram.
        """

        def __init__(self, out_dim: int = 128, dropout: float = 0.25):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1: (1, 64, 32) -> (32, 32, 16)
                nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(dropout * 0.5),
                # Block 2: (32, 32, 16) -> (64, 16, 8)
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(dropout * 0.5),
                # Block 3: (64, 16, 8) -> (128, 4, 4)
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((2, 2)),
            )
            self.proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 2 * 2, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(self.features(x))


    class PatientAudioEncoder(nn.Module):
        """
        Aggregates all K solicited coughs of a single patient:
          1. Applies `CoughCNN2D` to each cough clip -> (B, K, D_cnn)
          2. Passes cough sequence through a Bidirectional LSTM -> (B, K, 2 * H_lstm)
          3. Uses masked additive attention pooling to weight informative coughs and produce
             a single patient-level audio representation `h_audio` of shape (B, out_dim)
             as well as per-cough sequence tokens for cross-modal attention.
        """

        def __init__(
            self,
            cnn_dim: int = 128,
            lstm_hidden: int = 64,
            out_dim: int = 128,
            dropout: float = 0.25,
        ):
            super().__init__()
            self.cough_cnn = CoughCNN2D(out_dim=cnn_dim, dropout=dropout)
            self.bilstm = nn.LSTM(
                input_size=cnn_dim,
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            bilstm_out_dim = lstm_hidden * 2
            self.attn_scorer = nn.Sequential(
                nn.Linear(bilstm_out_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
            self.out_proj = nn.Sequential(
                nn.Linear(bilstm_out_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            )

        def forward(self, coughs: torch.Tensor, cough_mask: torch.Tensor):
            """
            Args:
                coughs: (B, K, 1, 64, 32)
                cough_mask: (B, K) float/bool mask where 1 = valid cough, 0 = padding
            Returns:
                patient_audio_emb: (B, out_dim)
                cough_tokens: (B, K, out_dim)
                attn_weights: (B, K)
            """
            bsz, k_coughs, ch, n_mels, n_frames = coughs.shape
            flat_coughs = coughs.reshape(bsz * k_coughs, ch, n_mels, n_frames)
            cough_vecs = self.cough_cnn(flat_coughs).reshape(bsz, k_coughs, -1)

            lstm_out, _ = self.bilstm(cough_vecs)  # (B, K, 2*H)
            attn_logits = self.attn_scorer(lstm_out).squeeze(-1)  # (B, K)

            # Mask out padded cough slots before softmax
            attn_logits = attn_logits.masked_fill(cough_mask <= 0.0, -1e4)
            attn_weights = F.softmax(attn_logits, dim=-1)  # (B, K)

            pooled = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, 2*H)
            patient_audio_emb = self.out_proj(pooled)
            cough_tokens = self.out_proj(lstm_out)
            return patient_audio_emb, cough_tokens, attn_weights


    class PretrainedAudioEncoderWrapper(nn.Module):
        """
        Comparison backbone (Section 4.2.1): Wraps a deeper residual spectrogram encoder
        (or torchaudio pretrained backbone) with attention pooling over patient coughs.
        """

        def __init__(self, out_dim: int = 128, dropout: float = 0.25):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU(),
            )
            self.res1 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
            )
            self.down = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(32, 96, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(96),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(96, out_dim),
                nn.LayerNorm(out_dim),
                nn.Dropout(dropout),
            )

        def forward(self, coughs: torch.Tensor, cough_mask: torch.Tensor):
            bsz, k_coughs, ch, h, w = coughs.shape
            x = coughs.reshape(bsz * k_coughs, ch, h, w)
            s = self.stem(x)
            x = F.gelu(s + self.res1(s))
            vecs = self.down(x).reshape(bsz, k_coughs, -1)
            mask = cough_mask.unsqueeze(-1)
            pooled = (vecs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            weights = cough_mask / cough_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            return pooled, vecs, weights
else:
    class CoughCNN2D:
        pass

    class PatientAudioEncoder:
        pass

    class PretrainedAudioEncoderWrapper:
        pass
