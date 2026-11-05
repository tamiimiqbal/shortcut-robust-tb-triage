"""
Classical & Deep Baselines + Shortcut-Invariant Representation Learner (Section 4.3).

Automatically uses `scikit-learn` / `xgboost` when installed (e.g., on Google Colab)
and provides deterministic SciPy/NumPy optimization fallbacks when run in minimal sandboxes.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class ClinicalLogisticRegression:
    """L2-regularized Class-Weighted Logistic Regression for Clinical Features."""

    def __init__(self, c_reg: float = 1.0):
        self.c_reg = c_reg
        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.latents_proj_: Optional[np.ndarray] = None
        self._sk_model = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "ClinicalLogisticRegression":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(42)
        self.latents_proj_ = rng.normal(0, 1.0 / np.sqrt(X.shape[1]), size=(X.shape[1], 32))

        if SKLEARN_AVAILABLE:
            self._sk_model = LogisticRegression(C=self.c_reg, class_weight="balanced", max_iter=500, random_state=42)
            self._sk_model.fit(X, y.astype(int), sample_weight=sample_weight)
            return self

        # Pure SciPy L-BFGS-B optimization with class balance
        pos_w = (len(y) - y.sum()) / max(y.sum(), 1.0)
        weights = np.where(y > 0.5, pos_w, 1.0)
        if sample_weight is not None:
            weights = weights * sample_weight

        d = X.shape[1]
        def obj(params):
            w, b = params[:d], params[d]
            z = X @ w + b
            p = np.clip(_sigmoid(z), 1e-7, 1.0 - 1e-7)
            loss = -np.mean(weights * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
            reg = 0.5 * (1.0 / self.c_reg) * np.sum(w ** 2) / len(y)
            grad_z = weights * (p - y) / len(y)
            grad_w = X.T @ grad_z + (1.0 / self.c_reg) * w / len(y)
            grad_b = np.sum(grad_z)
            return loss + reg, np.append(grad_w, grad_b)

        res = minimize(obj, x0=np.zeros(d + 1), jac=True, method="L-BFGS-B")
        self.w_ = res.x[:d]
        self.b_ = float(res.x[d])
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self._sk_model is not None:
            probs = np.clip(self._sk_model.predict_proba(X)[:, 1], 1e-6, 1.0 - 1e-6)
            return np.log(probs / (1.0 - probs))
        return (X @ self.w_ + self.b_).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.predict_logits(X)).astype(np.float32)

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(np.asarray(X, dtype=np.float64) @ self.latents_proj_).astype(np.float32)


class ClinicalRandomForest:
    """Random Forest Classifier on Clinical Questionnaire Features."""

    def __init__(self, n_estimators: int = 150, max_depth: int = 6, seed: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self._sk_model = None
        self._ensemble_weights = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClinicalRandomForest":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if SKLEARN_AVAILABLE:
            self._sk_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight="balanced",
                random_state=self.seed,
            )
            self._sk_model.fit(X, y.astype(int))
            return self

        # Bagged random feature subspace non-linear threshold trees fallback
        rng = np.random.default_rng(self.seed)
        self._ensemble_weights = []
        n, d = X.shape
        for _ in range(40):
            boot_idx = rng.choice(n, size=n, replace=True)
            feat_idx = rng.choice(d, size=max(4, int(np.sqrt(d)) * 2), replace=False)
            X_sub = np.column_stack([X[boot_idx][:, feat_idx], X[boot_idx][:, feat_idx] ** 2])
            lr = ClinicalLogisticRegression(c_reg=2.0).fit(X_sub, y[boot_idx])
            self._ensemble_weights.append((feat_idx, lr))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self._sk_model is not None:
            return self._sk_model.predict_proba(X)[:, 1].astype(np.float32)
        preds = []
        for feat_idx, lr in self._ensemble_weights:
            X_sub = np.column_stack([X[:, feat_idx], X[:, feat_idx] ** 2])
            preds.append(lr.predict_proba(X_sub))
        return np.mean(preds, axis=0).astype(np.float32)

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        p = np.clip(self.predict_proba(X), 1e-5, 1.0 - 1e-5)
        return np.log(p / (1.0 - p)).astype(np.float32)


class ClinicalXGBoost:
    """Gradient Boosted Decision Trees (XGBoost) on Clinical Questionnaire Features."""

    def __init__(self, n_estimators: int = 120, learning_rate: float = 0.05, seed: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.seed = seed
        self._xgb = None
        self._stages = []
        self._base_logit = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClinicalXGBoost":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if XGB_AVAILABLE:
            pos_weight = float((len(y) - y.sum()) / max(y.sum(), 1.0))
            self._xgb = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=4,
                learning_rate=self.learning_rate,
                scale_pos_weight=pos_weight,
                eval_metric="logloss",
                random_state=self.seed,
            )
            self._xgb.fit(X, y.astype(int))
            return self

        # Gradient boosting over second-order polynomial feature interactions
        rng = np.random.default_rng(self.seed)
        p_mean = np.clip(y.mean(), 0.05, 0.95)
        self._base_logit = float(np.log(p_mean / (1.0 - p_mean)))
        cur_logits = np.full(len(y), self._base_logit)
        self._stages = []
        d = X.shape[1]
        for _ in range(25):
            probs = _sigmoid(cur_logits)
            residuals = y - probs
            f_idx = rng.choice(d, size=min(8, d), replace=False)
            basis = np.column_stack([X[:, f_idx], np.tanh(X[:, f_idx])])
            w, _, _, _ = np.linalg.lstsq(basis, residuals, rcond=None)
            cur_logits += 0.25 * (basis @ w)
            self._stages.append((f_idx, w))
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self._xgb is not None:
            p = np.clip(self._xgb.predict_proba(X)[:, 1], 1e-5, 1.0 - 1e-5)
            return np.log(p / (1.0 - p)).astype(np.float32)
        logits = np.full(len(X), self._base_logit)
        for f_idx, w in self._stages:
            basis = np.column_stack([X[:, f_idx], np.tanh(X[:, f_idx])])
            logits += 0.25 * (basis @ w)
        return logits.astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.predict_logits(X)).astype(np.float32)


class AudioMFCCRandomForest:
    """Classical Audio Baseline: Aggregated Patient MFCC Statistics + Random Forest."""

    def __init__(self, seed: int = 42):
        self.rf = ClinicalRandomForest(n_estimators=150, max_depth=7, seed=seed)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    @staticmethod
    def aggregate_patient_mfccs(df, patient_mfccs: Dict[str, np.ndarray]) -> np.ndarray:
        feats = []
        for pid in df["participant"]:
            mat = patient_mfccs[pid]  # (K, 40)
            f_vec = np.concatenate([
                mat.mean(axis=0),
                mat.std(axis=0),
                mat.max(axis=0),
                mat.min(axis=0),
            ])
            feats.append(f_vec)
        return np.vstack(feats).astype(np.float32)

    def fit(self, X_mfcc: np.ndarray, y: np.ndarray) -> "AudioMFCCRandomForest":
        self.mean_ = X_mfcc.mean(axis=0, keepdims=True)
        self.std_ = X_mfcc.std(axis=0, keepdims=True) + 1e-6
        X_norm = (X_mfcc - self.mean_) / self.std_
        self.rf.fit(X_norm, y)
        return self

    def predict_proba(self, X_mfcc: np.ndarray) -> np.ndarray:
        X_norm = (X_mfcc - self.mean_) / self.std_
        return self.rf.predict_proba(X_norm)

    def predict_logits(self, X_mfcc: np.ndarray) -> np.ndarray:
        X_norm = (X_mfcc - self.mean_) / self.std_
        return self.rf.predict_logits(X_norm)
