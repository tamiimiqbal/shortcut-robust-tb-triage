"""
Detection & Calibration Metrics for Tuberculosis Triage (Section 4.4):
  - AUROC & full ROC curve
  - Sensitivity, Specificity, and Specificity at >=90% Sensitivity (WHO Triage Target)
  - Expected Calibration Error (ECE) & Reliability Diagram bins
"""

from typing import Dict, Tuple
import numpy as np


def compute_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FPR, TPR, and thresholds for binary classification."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    desc_idx = np.argsort(-y_prob)
    y_true_sorted = y_true[desc_idx]
    y_prob_sorted = y_prob[desc_idx]

    distinct_indices = np.where(np.diff(y_prob_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0.0, tps]
    fps = np.r_[0.0, fps]
    thresholds = np.r_[y_prob_sorted[0] + 1.0, y_prob_sorted[threshold_idxs]]

    p_total = max(tps[-1], 1.0)
    n_total = max(fps[-1], 1.0)
    tpr = tps / p_total
    fpr = fps / n_total
    return fpr, tpr, thresholds


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (AUROC)."""
    fpr, tpr, _ = compute_roc_curve(y_true, y_prob)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))


def compute_who_triage_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float = 0.90,
    target_specificity: float = 0.70,
) -> Dict[str, float]:
    """
    Evaluate model against WHO Triage Target (>=90% sensitivity, >=70% specificity).
    Returns:
      - auroc
      - spec_at_90_sens: Highest specificity achieved where sensitivity >= 90%
      - sens_at_target: Actual sensitivity at that threshold
      - threshold_90_sens: Operating probability threshold for 90% sensitivity
      - who_gap: Shortfall from 70% specificity target (0.0 if spec >= 0.70)
    """
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_prob)
    specificities = 1.0 - fpr
    auroc = compute_auroc(y_true, y_prob)

    valid_mask = tpr >= target_sensitivity
    if np.any(valid_mask):
        best_idx = np.where(valid_mask)[0][np.argmax(specificities[valid_mask])]
        spec_at_90 = float(specificities[best_idx])
        sens_at_90 = float(tpr[best_idx])
        thresh_90 = float(thresholds[best_idx])
    else:
        spec_at_90 = 0.0
        sens_at_90 = float(tpr[-1])
        thresh_90 = 0.0

    who_gap = max(0.0, target_specificity - spec_at_90)
    meets_who = bool(spec_at_90 >= target_specificity and sens_at_90 >= target_sensitivity)

    return {
        "auroc": round(auroc, 4),
        "spec_at_90_sens": round(spec_at_90, 4),
        "sens_at_90_thresh": round(sens_at_90, 4),
        "threshold_90_sens": round(float(np.clip(thresh_90, 0.01, 0.99)), 4),
        "who_spec_gap": round(who_gap, 4),
        "meets_who_target": meets_who,
    }


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Compute Expected Calibration Error (ECE) and bin statistics for reliability diagrams.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    bin_Boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_Boundaries[:-1]
    bin_uppers = bin_Boundaries[1:]

    ece = 0.0
    bin_accs = np.zeros(n_bins, dtype=np.float64)
    bin_confs = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    n_total = max(len(y_true), 1)
    for i, (b_low, b_up) in enumerate(zip(bin_lowers, bin_uppers)):
        if i == 0:
            in_bin = (y_prob >= b_low) & (y_prob <= b_up)
        else:
            in_bin = (y_prob > b_low) & (y_prob <= b_up)
        count = int(np.sum(in_bin))
        bin_counts[i] = count
        if count > 0:
            acc = float(np.mean(y_true[in_bin]))
            conf = float(np.mean(y_prob[in_bin]))
            bin_accs[i] = acc
            bin_confs[i] = conf
            ece += (count / n_total) * abs(acc - conf)
        else:
            bin_accs[i] = 0.5 * (b_low + b_up)
            bin_confs[i] = 0.5 * (b_low + b_up)

    return float(ece), {
        "bin_centers": 0.5 * (bin_lowers + bin_uppers),
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
    }
