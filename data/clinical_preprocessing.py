"""
Clinical Questionnaire Preprocessing Pipeline (Section 4.1).

Performs:
- One-hot / binary encoding of categorical variables (sex, fever, weight_loss, night_sweats, hemoptysis, smoke_lweek, prior_tb)
- Z-score normalization of numeric variables (age, height, weight, bmi, reported_cough_dur, heart_rate, temperature)
  fit strictly on training split
- Explicit binary missing-value indicator columns (`<col>_missing`) for every feature
"""

from typing import List, Optional
import numpy as np
import pandas as pd


class ClinicalPreprocessor:
    NUMERIC_COLS: List[str] = [
        "age",
        "height",
        "weight",
        "bmi",
        "reported_cough_dur",
        "heart_rate",
        "temperature",
    ]
    CATEGORICAL_COLS: List[str] = [
        "sex",
        "fever",
        "weight_loss",
        "night_sweats",
        "hemoptysis",
        "smoke_lweek",
        "prior_tb",
    ]

    def __init__(self):
        self.means_: dict = {}
        self.stds_: dict = {}
        self.modes_: dict = {}
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "ClinicalPreprocessor":
        for col in self.NUMERIC_COLS:
            vals = pd.to_numeric(df[col], errors="coerce")
            mean_val = float(vals.mean()) if not np.isnan(vals.mean()) else 0.0
            std_val = float(vals.std())
            if np.isnan(std_val) or std_val < 1e-6:
                std_val = 1.0
            self.means_[col] = mean_val
            self.stds_[col] = std_val

        for col in self.CATEGORICAL_COLS:
            if col == "sex":
                self.modes_[col] = 1.0 if (df[col].astype(str).str.lower() == "male").mean() >= 0.5 else 0.0
            else:
                vals = pd.to_numeric(df[col], errors="coerce")
                self.modes_[col] = float(vals.median()) if not np.isnan(vals.median()) else 0.0

        self.feature_names_ = (
            [f"{c}_norm" for c in self.NUMERIC_COLS]
            + [f"{c}_enc" for c in self.CATEGORICAL_COLS]
            + [f"{c}_missing" for c in (self.NUMERIC_COLS + self.CATEGORICAL_COLS)]
        )
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("ClinicalPreprocessor must be fitted on training data before transform().")

        n = len(df)
        num_feats = []
        cat_feats = []
        missing_feats = []

        for col in self.NUMERIC_COLS:
            raw = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
            miss = np.isnan(raw).astype(np.float32)
            filled = np.where(np.isnan(raw), self.means_[col], raw)
            normed = (filled - self.means_[col]) / self.stds_[col]
            num_feats.append(normed)
            missing_feats.append(miss)

        for col in self.CATEGORICAL_COLS:
            if col == "sex":
                s_str = df[col].astype(str).str.strip().str.lower()
                miss = (~s_str.isin(["male", "female", "m", "f", "1", "0"])).to_numpy(dtype=np.float32)
                enc = np.where(s_str.isin(["male", "m", "1"]), 1.0, 0.0).astype(np.float32)
                enc = np.where(miss > 0.5, self.modes_[col], enc)
            else:
                raw = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
                miss = np.isnan(raw).astype(np.float32)
                enc = np.where(np.isnan(raw), self.modes_[col], raw).astype(np.float32)
            cat_feats.append(enc)
            missing_feats.append(miss)

        mat = np.column_stack(num_feats + cat_feats + missing_feats).astype(np.float32)
        return mat

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
