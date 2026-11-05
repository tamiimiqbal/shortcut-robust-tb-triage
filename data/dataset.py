"""
PyTorch Dataset & Collate Function for Patient-Level Multimodal TB Triage.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


class CODAMultimodalDataset(Dataset):
    """
    Patient-level dataset yielding:
      - `coughs`: Tensor of shape (K_max, 1, 64, 32)
      - `cough_mask`: Boolean/float mask of shape (K_max,) indicating valid coughs
      - `clinical`: Tensor of shape (D_clin,)
      - `tb_label`: float scalar in {0.0, 1.0}
      - `site_label`: int64 scalar in {0, ..., 6}
      - `sample_weight`: float scalar for site-balanced resampling
    """

    def __init__(
        self,
        df: pd.DataFrame,
        clinical_features: np.ndarray,
        patient_spectrograms: Dict[str, np.ndarray],
        max_coughs: int = 6,
        sample_weights: np.ndarray = None,
    ):
        self.df = df.reset_index(drop=True)
        self.clinical_features = clinical_features.astype(np.float32)
        self.patient_spectrograms = patient_spectrograms
        self.max_coughs = max_coughs
        if sample_weights is None:
            self.sample_weights = np.ones(len(self.df), dtype=np.float32)
        else:
            self.sample_weights = sample_weights.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        row = self.df.iloc[idx]
        pid = row["participant"]
        specs = self.patient_spectrograms[pid]  # (K, 64, 32)
        k_actual = min(len(specs), self.max_coughs)

        padded_specs = np.zeros((self.max_coughs, 1, specs.shape[1], specs.shape[2]), dtype=np.float32)
        padded_specs[:k_actual, 0] = specs[:k_actual]
        cough_mask = np.zeros(self.max_coughs, dtype=np.float32)
        cough_mask[:k_actual] = 1.0

        item = {
            "participant": pid,
            "coughs": padded_specs,
            "cough_mask": cough_mask,
            "clinical": self.clinical_features[idx],
            "tb_label": np.float32(row["tb_status"]),
            "site_label": np.int64(row["site_id"]),
            "country": row["country"],
            "sample_weight": np.float32(self.sample_weights[idx]),
        }
        if TORCH_AVAILABLE:
            return {
                "participant": pid,
                "coughs": torch.from_numpy(padded_specs),
                "cough_mask": torch.from_numpy(cough_mask),
                "clinical": torch.from_numpy(self.clinical_features[idx]),
                "tb_label": torch.tensor(row["tb_status"], dtype=torch.float32),
                "site_label": torch.tensor(row["site_id"], dtype=torch.long),
                "country": row["country"],
                "sample_weight": torch.tensor(self.sample_weights[idx], dtype=torch.float32),
            }
        return item


def collate_patient_batch(batch: List[Dict]) -> Dict:
    """Collate list of patient records into a batched dictionary."""
    if TORCH_AVAILABLE and isinstance(batch[0]["coughs"], torch.Tensor):
        return {
            "participant": [b["participant"] for b in batch],
            "country": [b["country"] for b in batch],
            "coughs": torch.stack([b["coughs"] for b in batch], dim=0),
            "cough_mask": torch.stack([b["cough_mask"] for b in batch], dim=0),
            "clinical": torch.stack([b["clinical"] for b in batch], dim=0),
            "tb_label": torch.stack([b["tb_label"] for b in batch], dim=0),
            "site_label": torch.stack([b["site_label"] for b in batch], dim=0),
            "sample_weight": torch.stack([b["sample_weight"] for b in batch], dim=0),
        }
    return {
        "participant": [b["participant"] for b in batch],
        "country": [b["country"] for b in batch],
        "coughs": np.stack([b["coughs"] for b in batch], axis=0),
        "cough_mask": np.stack([b["cough_mask"] for b in batch], axis=0),
        "clinical": np.stack([b["clinical"] for b in batch], axis=0),
        "tb_label": np.stack([b["tb_label"] for b in batch], axis=0),
        "site_label": np.stack([b["site_label"] for b in batch], axis=0),
        "sample_weight": np.stack([b["sample_weight"] for b in batch], axis=0),
    }
