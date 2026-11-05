"""
Patient-Level Cross-Validation, Leave-One-Country-Out (LOCO) Splits,
and Site-Balanced Resampling (Section 4.1 & 4.4).
"""

from typing import Dict, Iterator, List, Tuple
import numpy as np
import pandas as pd


def create_patient_kfold_splits(
    clinical_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create patient-level stratified k-fold splits stratified jointly by (country, tb_status)
    so no patient's coughs ever appear in both train and test sets.
    """
    rng = np.random.default_rng(seed)
    df = clinical_df.reset_index(drop=True)
    strata = df["country"].astype(str) + "_" + df["tb_status"].astype(str)

    fold_Assignments = np.zeros(len(df), dtype=int)
    for stratum_val in strata.unique():
        idx = np.where(strata == stratum_val)[0]
        rng.shuffle(idx)
        for i, sample_idx in enumerate(idx):
            fold_Assignments[sample_idx] = i % n_splits

    splits = []
    for fold in range(n_splits):
        test_idx = np.where(fold_Assignments == fold)[0]
        train_idx = np.where(fold_Assignments != fold)[0]
        splits.append((train_idx, test_idx))
    return splits


def create_loco_splits(
    clinical_df: pd.DataFrame,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Leave-One-Country-Out (LOCO) protocol:
    For each country c in CODA, train on the other 6 countries and test on country c.
    """
    df = clinical_df.reset_index(drop=True)
    loco_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for country in sorted(df["country"].unique()):
        test_idx = np.where(df["country"] == country)[0]
        train_idx = np.where(df["country"] != country)[0]
        loco_dict[country] = (train_idx, test_idx)
    return loco_dict


def compute_site_balanced_weights(
    clinical_df_train: pd.DataFrame,
) -> np.ndarray:
    """
    Compute sample weights inversely proportional to joint (country, tb_status) cell counts.
    Ensures every country has balanced effective TB+/TB- weight during training, breaking
    the country base-rate shortcut (Section 4.1).
    """
    strata = clinical_df_train["country"].astype(str) + "_" + clinical_df_train["tb_status"].astype(str)
    counts = strata.value_counts().to_dict()
    total = float(len(clinical_df_train))
    n_cells = float(len(counts))
    weights = np.array(
        [total / (n_cells * max(1, counts[s])) for s in strata],
        dtype=np.float32,
    )
    return weights / weights.mean()
