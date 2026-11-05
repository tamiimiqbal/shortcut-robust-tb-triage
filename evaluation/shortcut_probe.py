"""
Shortcut Diagnostics (Section 3 RQ2 & Section 4.4):
  1. `evaluate_site_linear_probe`: Predict recording country (7 countries, chance = 14.3%)
     from frozen learned representations using a regularized linear probe.
  2. `evaluate_country_tb_rate_shortcut`: Regress mean predicted TB probability (on TB-negative
     patients) against each country's ground-truth TB prevalence rate (slope & R^2).
"""

from typing import Dict
import numpy as np
import pandas as pd


def evaluate_site_linear_probe(
    features_train: np.ndarray,
    sites_train: np.ndarray,
    features_test: np.ndarray,
    sites_test: np.ndarray,
    n_sites: int = 7,
    l2_reg: float = 1.0,
) -> Dict[str, float]:
    """
    Fit a multinomial ridge/linear classifier on frozen learned features to predict
    country/site label `s_i in {0, ..., 6}`.
    Lower accuracy (closer to chance = 1/7 = 14.29%) indicates stronger site invariance.
    """
    X_tr = np.asarray(features_train, dtype=np.float64)
    s_tr = np.asarray(sites_train, dtype=np.int64).ravel()
    X_te = np.asarray(features_test, dtype=np.float64)
    s_te = np.asarray(sites_test, dtype=np.int64).ravel()

    # Standardize features on train split
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-6
    X_tr_n = np.column_stack([ (X_tr - mu) / sd, np.ones(len(X_tr)) ])
    X_te_n = np.column_stack([ (X_te - mu) / sd, np.ones(len(X_te)) ])

    # One-hot target matrix
    Y_tr = np.eye(n_sites)[s_tr]
    d = X_tr_n.shape[1]
    reg_mat = l2_reg * np.eye(d)
    reg_mat[-1, -1] = 0.0  # Do not regularize bias

    W = np.linalg.solve(X_tr_n.T @ X_tr_n + reg_mat, X_tr_n.T @ Y_tr)
    logits_te = X_te_n @ W
    preds_te = np.argmax(logits_te, axis=1)
    acc = float(np.mean(preds_te == s_te))
    chance = 1.0 / float(n_sites)

    return {
        "probe_accuracy": round(acc, 4),
        "chance_level": round(chance, 4),
        "excess_over_chance": round(max(0.0, acc - chance), 4),
    }


def evaluate_country_tb_rate_shortcut(
    clinical_df: pd.DataFrame,
    y_prob: np.ndarray,
) -> Dict[str, object]:
    """
    Replicates the 2026 shortcut diagnostic (Section 2 & 4.4):
    For each country, compute:
      - `true_tb_rate`: Ground-truth TB-positive proportion in that country
      - `mean_pred_prob_tb_neg`: Mean predicted TB probability among TB-negative patients
    Then compute the linear regression slope and R^2.
    A shortcut-heavy model has slope ~1.20 and R^2 ~0.95; a site-invariant model has slope -> 0.
    """
    df = clinical_df.copy().reset_index(drop=True)
    df["pred_prob"] = np.asarray(y_prob, dtype=np.float64).ravel()

    country_stats = []
    for country, grp in df.groupby("country"):
        true_rate = float(grp["tb_status"].mean())
        neg_grp = grp[grp["tb_status"] == 0]
        mean_pred_neg = float(neg_grp["pred_prob"].mean()) if len(neg_grp) > 0 else float(grp["pred_prob"].mean())
        mean_pred_all = float(grp["pred_prob"].mean())
        country_stats.append({
            "country": country,
            "true_tb_rate": true_rate,
            "mean_pred_tb_neg": mean_pred_neg,
            "mean_pred_all": mean_pred_all,
        })

    stats_df = pd.DataFrame(country_stats).sort_values("true_tb_rate").reset_index(drop=True)
    x = stats_df["true_tb_rate"].to_numpy()
    y = stats_df["mean_pred_tb_neg"].to_numpy()

    x_mean, y_mean = x.mean(), y.mean()
    ss_xx = np.sum((x - x_mean) ** 2) + 1e-12
    ss_yy = np.sum((y - y_mean) ** 2) + 1e-12
    ss_xy = np.sum((x - x_mean) * (y - y_mean))

    slope = float(ss_xy / ss_xx)
    intercept = float(y_mean - slope * x_mean)
    r_squared = float((ss_xy ** 2) / (ss_xx * ss_yy))

    return {
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "r_squared": round(r_squared, 4),
        "country_table": stats_df,
    }
