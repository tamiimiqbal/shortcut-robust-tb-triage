"""
Publication-Grade Visualization Suite for CSE 414 TB Triage Project.
Generates 4 multi-panel diagnostic figures covering RQ1-RQ4.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# Cohesive publication color palette
PALETTE = {
    "navy": "#1B365D",
    "teal": "#008080",
    "coral": "#E05A47",
    "amber": "#D99B26",
    "slate": "#5C6B73",
    "light_bg": "#F8FAFC",
    "green": "#2E7D32",
}


def generate_all_publication_figures(
    results_bundle: Dict,
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#F9FBFD",
        "axes.edgecolor": "#CBD5E1",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })

    paths = {}

    # -------------------------------------------------------------------------
    # Figure 1: LOCO Generalization Across Unseen Countries (RQ1, RQ3, RQ4)
    # -------------------------------------------------------------------------
    loco_df: pd.DataFrame = results_bundle["loco_country_df"]
    summary_df: pd.DataFrame = results_bundle["summary_table"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=220)
    countries = loco_df["country"].tolist()
    x = np.arange(len(countries))
    w = 0.25

    ax1.bar(x - w, loco_df["audio_only_auroc"], width=w, label="Audio CNN-BiLSTM", color=PALETTE["slate"], alpha=0.85)
    ax1.bar(x, loco_df["late_fusion_auroc"], width=w, label="Standard Late-Fusion", color=PALETTE["amber"], alpha=0.90)
    ax1.bar(x + w, loco_df["proposed_auroc"], width=w, label="Proposed (Gated + GRL)", color=PALETTE["teal"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(countries, rotation=22, ha="right")
    ax1.set_ylim(0.55, 0.95)
    ax1.set_ylabel("Held-Out Country AUROC")
    ax1.set_title("A. Leave-One-Country-Out (LOCO) AUROC by Unseen Country")
    ax1.legend(loc="lower right", frameon=True, facecolor="white")

    # Panel B: Specificity at 90% Sensitivity vs WHO 70% Target
    models_short = summary_df["Model"].tolist()
    y_pos = np.arange(len(models_short))
    specs_90 = summary_df["LOCO_Spec_at_90Sens"].to_numpy() * 100.0
    colors_bar = [
        PALETTE["slate"] if "Proposed" not in m else PALETTE["teal"]
        for m in models_short
    ]
    bars = ax2.barh(y_pos, specs_90, color=colors_bar, height=0.58)
    ax2.axvline(70.0, color=PALETTE["coral"], linestyle="--", linewidth=2.0, label="WHO Triage Target (70% Spec @ 90% Sens)")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models_short)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 88)
    ax2.set_xlabel("Specificity at 90% Sensitivity (%) on Unseen Countries")
    ax2.set_title("B. WHO Triage Benchmark (Specificity @ >=90% Sensitivity)")
    for bar, val in zip(bars, specs_90):
        ax2.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax2.legend(loc="center right", frameon=True, facecolor="white", fontsize=8.8)

    fig.tight_layout(pad=1.8)
    f1_path = output_dir / "fig1_loco_auroc_and_who_target.png"
    fig.savefig(f1_path, bbox_inches="tight")
    plt.close(fig)
    paths["fig1"] = f1_path

    # -------------------------------------------------------------------------
    # Figure 2: Site Shortcut Probe & Country TB-Rate Regression (RQ2 & RQ3)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=220)

    probe_models = ["Audio MFCC+RF", "Audio CNN-BiLSTM", "Clinical MLP", "Late-Fusion (No GRL)", "Proposed (GRL + Balanced)"]
    probe_accs = results_bundle["probe_accuracies_pct"]
    bar_cols = [PALETTE["coral"], PALETTE["coral"], PALETTE["slate"], PALETTE["amber"], PALETTE["teal"]]

    x_pr = np.arange(len(probe_models))
    b1 = ax1.bar(x_pr, probe_accs, color=bar_cols, width=0.52)
    ax1.axhline(14.29, color=PALETTE["navy"], linestyle="--", linewidth=1.8, label="Random Chance (1/7 Countries = 14.3%)")
    ax1.set_ylabel("Country Linear Probe Accuracy (%)")
    ax1.set_ylim(0, 92)
    ax1.set_xticks(x_pr)
    ax1.set_xticklabels(probe_models, rotation=18, ha="right")
    ax1.set_title("A. Site Information Leakage in Learned Features (Lower = Better)")
    for bar, val in zip(b1, probe_accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1.8, f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white")

    # Panel B: Country TB Rate vs Mean Predicted Probability on TB-Negative Patients
    shortcut_late = results_bundle["shortcut_late_fusion"]
    shortcut_prop = results_bundle["shortcut_proposed"]
    df_late_c = shortcut_late["country_table"]
    df_prop_c = shortcut_prop["country_table"]

    x_rates = df_late_c["true_tb_rate"].to_numpy()
    y_late = df_late_c["mean_pred_tb_neg"].to_numpy()
    y_prop = df_prop_c["mean_pred_tb_neg"].to_numpy()

    ax2.scatter(x_rates * 100, y_late * 100, color=PALETTE["coral"], s=65, zorder=4,
                label=f"Standard Late-Fusion (Slope={shortcut_late['slope']:.2f}, R²={shortcut_late['r_squared']:.2f})")
    ax2.scatter(x_rates * 100, y_prop * 100, color=PALETTE["teal"], s=65, marker="s", zorder=4,
                label=f"Proposed GRL Model (Slope={shortcut_prop['slope']:.2f}, R²={shortcut_prop['r_squared']:.2f})")

    x_line = np.linspace(x_rates.min(), x_rates.max(), 50)
    ax2.plot(x_line * 100, (shortcut_late["slope"] * x_line + shortcut_late["intercept"]) * 100,
             color=PALETTE["coral"], linestyle="-", linewidth=2.0)
    ax2.plot(x_line * 100, (shortcut_prop["slope"] * x_line + shortcut_prop["intercept"]) * 100,
             color=PALETTE["teal"], linestyle="-", linewidth=2.0)

    for _, r in df_late_c.iterrows():
        ax2.annotate(r["country"][:3], (r["true_tb_rate"] * 100 + 0.4, r["mean_pred_tb_neg"] * 100 + 0.6),
                     fontsize=8, color=PALETTE["slate"])

    ax2.set_xlabel("Country Ground-Truth TB Prevalence Rate (%)")
    ax2.set_ylabel("Mean Predicted TB Prob. on TB-Negative Cases (%)")
    ax2.set_title("B. Shortcut Diagnosis: Predictions vs. Country TB Rate (TB- Cases)")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.8)

    fig.tight_layout(pad=1.8)
    f2_path = output_dir / "fig2_shortcut_analysis_probe_and_slope.png"
    fig.savefig(f2_path, bbox_inches="tight")
    plt.close(fig)
    paths["fig2"] = f2_path

    # -------------------------------------------------------------------------
    # Figure 3: ROC Curves with WHO Target Zone & Reliability Diagram (Calibration)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=220)

    roc_dict = results_bundle["roc_curves"]
    for label, (fpr, tpr, auroc_val, col, ls) in roc_dict.items():
        ax1.plot(fpr, tpr, label=f"{label} (AUROC = {auroc_val:.3f})", color=col, linestyle=ls, linewidth=2.2)

    # Highlight WHO Triage Target box (FPR <= 0.30, TPR >= 0.90)
    ax1.fill_between([0.0, 0.30], 0.90, 1.0, color=PALETTE["green"], alpha=0.18, label="WHO Target Zone (>=90% Sens, >=70% Spec)")
    ax1.scatter([0.30], [0.90], color=PALETTE["green"], s=80, marker="*", zorder=5)
    ax1.plot([0, 1], [0, 1], color="#94A3B8", linestyle=":", linewidth=1.2)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.02)
    ax1.set_xlabel("1 - Specificity (False Positive Rate)")
    ax1.set_ylabel("Sensitivity (True Positive Rate)")
    ax1.set_title("A. Held-Out Country ROC Curves & WHO Triage Zone")
    ax1.legend(loc="lower right", frameon=True, facecolor="white", fontsize=8.8)

    # Panel B: Reliability Diagram before & after Temperature Scaling
    cal_uncal = results_bundle["cal_uncalibrated"]
    cal_scaled = results_bundle["cal_temperature_scaled"]
    ax2.plot([0, 1], [0, 1], linestyle="--", color="#64748B", label="Perfect Calibration")
    ax2.plot(
        cal_uncal["bins"]["bin_confs"],
        cal_uncal["bins"]["bin_accs"],
        marker="o",
        color=PALETTE["coral"],
        linewidth=2.0,
        label=f"Before Temp. Scaling (ECE = {cal_uncal['ece']:.3f})",
    )
    ax2.plot(
        cal_scaled["bins"]["bin_confs"],
        cal_scaled["bins"]["bin_accs"],
        marker="s",
        color=PALETTE["teal"],
        linewidth=2.2,
        label=f"After Temp. Scaling T={cal_scaled['T']:.2f} (ECE = {cal_scaled['ece']:.3f})",
    )
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel("Mean Predicted TB Probability (Confidence)")
    ax2.set_ylabel("Observed Fraction of TB-Positive Patients")
    ax2.set_title("B. Reliability Diagram & Calibration Improvement (Section 4.2.6)")
    ax2.legend(loc="upper left", frameon=True, facecolor="white")

    fig.tight_layout(pad=1.8)
    f3_path = output_dir / "fig3_calibration_reliability_diagram.png"
    fig.savefig(f3_path, bbox_inches="tight")
    plt.close(fig)
    paths["fig3"] = f3_path

    # -------------------------------------------------------------------------
    # Figure 4: Ablation Study & Sample Solicited Cough Spectrogram Attention
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.2), dpi=220, gridspec_kw={"width_ratios": [1.15, 1.0]})

    ablation_df: pd.DataFrame = results_bundle["ablation_df"]
    y_ab = np.arange(len(ablation_df))
    ax1.barh(y_ab - 0.17, ablation_df["LOCO_AUROC"] * 100, height=0.34, label="LOCO AUROC (%)", color=PALETTE["navy"])
    ax1.barh(y_ab + 0.17, ablation_df["Spec_at_90Sens"] * 100, height=0.34, label="Spec @ 90% Sens (%)", color=PALETTE["teal"])
    ax1.set_yticks(y_ab)
    ax1.set_yticklabels(ablation_df["Variant"])
    ax1.invert_yaxis()
    ax1.set_xlim(25, 92)
    ax1.set_xlabel("Performance (%) on Unseen Countries (LOCO)")
    ax1.set_title("A. Ablation Study on Unseen Countries (LOCO)")
    ax1.legend(loc="lower right", frameon=True, facecolor="white")

    # Panel B: Sample Log-Mel Spectrogram of Solicited Cough with Attention Envelope
    sample_mel = results_bundle["sample_mel_spectrogram"]
    im = ax2.imshow(sample_mel, aspect="auto", origin="lower", cmap="magma", extent=[0.0, 0.5, 0, 64])
    ax2.set_xlabel("Time (seconds, 0.5s cough clip)")
    ax2.set_ylabel("Mel Frequency Bin (100 Hz – 6,000 Hz)")
    ax2.set_title("B. 0.5s Cough Log-Mel Spectrogram (TB+)")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Log Power (dB)")

    fig.tight_layout(pad=1.8)
    f4_path = output_dir / "fig4_ablation_and_cough_attention.png"
    fig.savefig(f4_path, bbox_inches="tight")
    plt.close(fig)
    paths["fig4"] = f4_path

    return paths
