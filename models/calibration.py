"""
Post-Hoc Temperature Scaling for Trustworthy Triage Probabilities (Section 4.2.6).
"""

import numpy as np
from scipy.optimize import minimize


class TemperatureScaler:
    """
    Fits temperature parameter T > 0 (and base-rate calibration offset b_0 to correct
    for class-weighted / focal-loss prior shift) on validation logits to minimize
    Negative Log-Likelihood (NLL) and lower Expected Calibration Error (ECE).
    """

    def __init__(self, init_temperature: float = 1.0):
        self.temperature: float = float(init_temperature)
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(val_logits, dtype=np.float64).ravel()
        labels = np.asarray(val_labels, dtype=np.float64).ravel()

        def nll_objective(params: np.ndarray) -> float:
            t = np.exp(params[0])
            b = params[1]
            scaled = (logits / max(t, 0.1)) + b
            probs = np.clip(self._sigmoid(scaled), 1e-7, 1.0 - 1e-7)
            nll = -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
            return float(nll + 1e-3 * (params[0] ** 2))

        res = minimize(nll_objective, x0=np.array([0.25, -0.5]), method="BFGS")
        self.temperature = float(np.clip(np.exp(res.x[0]), 1.15, 3.50))
        self.bias = float(res.x[1])
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        logits_arr = np.asarray(logits, dtype=np.float64)
        return self._sigmoid((logits_arr / self.temperature) + self.bias).astype(np.float32)
