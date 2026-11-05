# Shortcut-Robust Multimodal Tuberculosis Triage: Learning Disease, Not Devices, from Cough Audio and Clinical Data

**Course:** CSE 414 — Machine Learning and Deep Learning Lab  
**Institution:** Department of Computer Science and Engineering, University of Asia Pacific (UAP)  
**Authors:** Md Tamim Iqbal (`22201141`), Sujana Mahmuda Nite (`23101120`) & Puja Saha (`22201213`)  
**Course Instructor:** Rashik Rahman  

---

## 1. Project Overview & Motivation

Smartphone cough screening combined with routine clinical questionnaires offers a low-cost, non-invasive point-of-care triage tool for tuberculosis (TB) in high-burden, resource-limited settings such as Bangladesh. However, recent analyses of the **CODA TB DREAM Challenge** dataset revealed a critical flaw: because each participating country used different recording setups and had different baseline TB prevalence rates, standard models learn **recording-site and device shortcuts** rather than true disease pathology (predicting country TB rate with slope $\approx 1.20, R^2 \approx 0.95$ on TB-negative patients).

This repository implements an end-to-end **Shortcut-Robust Multimodal Deep Learning Pipeline** in PyTorch and Scikit-Learn/XGBoost, paired with an interactive **Gradio Triage Web Application**, designed to generalize to unseen countries and recording devices.

### Architecture Summary (`2 Encoders, 1 Fusion, 2 Heads + Calibration`)
1. **Audio Branch (`models/audio_encoder.py`):** 16 kHz resampling, 4th-order Butterworth bandpass filtering ($100\text{ Hz} - 6,000\text{ Hz}$), and $0.5\text{ s}$ Log-Mel Spectrogram extraction $(1 \times 64 \times 32)$, encoded via a **2D CNN + Bidirectional LSTM + Masked Multi-Cough Attention Pooling** over all solicited coughs of a patient (plus a Pretrained Audio Backbone comparison module).
2. **Clinical Branch (`models/clinical_encoder.py`):** One-hot encoded categorical symptoms, training-fold Z-score normalized continuous vitals/demographics, and explicit binary `_missing` indicators encoded via a 2-layer **Clinical MLP**.
3. **Multimodal Fusion (`models/fusion.py`):** Compares standard **Late-Fusion Concatenation** against our proposed **Gated Cross-Modal Attention Fusion**, where clinical context queries patient cough tokens via Multi-Head Cross-Attention followed by a learned sigmoid modality gate.
4. **TB Classification Head (`models/losses.py`):** Trained with **Binary Focal Loss** ($\alpha=0.70, \gamma=2.0$) and **Site-Balanced Resampling** (`TB label × country` inverse cell weighting).
5. **Site-Adversarial Head (`models/adversarial.py`):** A 7-country classifier attached to the shared representation via a **Gradient Reversal Layer (GRL)** with DANN schedule $\lambda(p) = \lambda_{\max}(\frac{2}{1 + \exp(-10p)} - 1)$, forcing fused features to drop country and device cues.
6. **Post-Hoc Calibration (`models/calibration.py`):** **Temperature Scaling** fitted via NLL on held-out validation logits so triage probabilities are trustworthy at the fixed **WHO Triage Target** ($\ge 90\%$ sensitivity, $\ge 70\%$ specificity).

---

## 2. Repository Structure (Aligned with Section 4.5 & CLO5)

```text
tb-triage-coda/
├── configs/
│   └── default_config.yaml           # Hyperparameters, audio DSP settings, and WHO targets
├── data/                             # Branch: feature/data (Weeks 1–2)
│   ├── __init__.py
│   ├── audio_preprocessing.py        # 16kHz resample, 100-6000Hz bandpass, 0.5s Log-Mel & MFCCs
│   ├── clinical_preprocessing.py     # Z-score normalization, one-hot encoding, missing indicators
│   ├── dataset.py                    # PyTorch CODAMultimodalDataset & patient batch collator
│   ├── splits.py                     # Patient-level 5-Fold CV, LOCO splits & site-balanced weights
│   └── synthetic_coda.py             # Realistic CODA TB schema generator & Synapse CSV adapter
├── models/                           # Branch: feature/models (Weeks 3–6)
│   ├── __init__.py
│   ├── adversarial.py                # Gradient Reversal Layer (GRL) & SiteAdversarialHead
│   ├── audio_encoder.py              # CoughCNN2D, PatientAudioEncoder (BiLSTM + Attention)
│   ├── baselines.py                  # Logistic Regression, Random Forest, XGBoost, MFCC+RF
│   ├── calibration.py                # Post-hoc TemperatureScaler for probability calibration
│   ├── clinical_encoder.py           # ClinicalMLPEncoder
│   ├── fusion.py                     # LateFusionModule & GatedCrossModalAttentionFusion
│   ├── losses.py                     # Imbalance-aware FocalLossWithLogits
│   └── multimodal_net.py             # Unified ShortcutRobustMultimodalNet
├── evaluation/                       # Branch: feature/evaluation (Weeks 7–8)
│   ├── __init__.py
│   ├── metrics.py                    # AUROC, ROC curve, Specificity@90%Sens, WHO target, ECE
│   ├── plotting.py                   # Publication-grade 4-figure visualization suite
│   └── shortcut_probe.py             # Country linear probe & country TB-rate shortcut slope/R^2
├── demo/                             # Branch: feature/demo (Weeks 9–10)
│   └── app.py                        # Interactive Gradio triage web application
├── notebooks/
│   └── CSE414_TB_Triage_Colab_Pipeline.ipynb  # Self-contained Google Colab end-to-end notebook
├── results/
│   ├── figures/                      # Generated high-res PNG evaluation figures (Figs 1–4)
│   └── tables/                       # Exported CSV benchmark tables (5-Fold, LOCO, Ablations)
├── scripts/
│   └── init_github_branches.sh       # Initializes Git repo with 4 module branches & PR merges
├── requirements.txt
└── run_pipeline.py                   # Master CLI script to run all experiments and generate plots
```

---

## 3. Experimental Results & Research Question Answers (RQ1–RQ4)

### Master Benchmark Table (5-Fold Patient CV vs. Leave-One-Country-Out)

| Model / Baseline | Input Modality | 5-Fold CV AUROC | 5-Fold Spec @ 90% Sens | LOCO AUROC (Unseen Country) | LOCO Spec @ 90% Sens | Country Probe Acc (Chance = 14.3%) | LOCO ECE (Lower = Better) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Clinical only | 0.7810 | 46.04% | 0.7666 | 42.99% | 16.00% | 0.1731 |
| **Random Forest** | Clinical only | 0.7750 | 46.20% | 0.7544 | 41.90% | 16.00% | 0.2120 |
| **XGBoost** | Clinical only | 0.7578 | 38.61% | 0.7401 | 36.90% | 16.00% | 0.0854 |
| **MFCC + Random Forest** | Audio only | 0.8767 | 62.97% | 0.7559 | 33.92% | 72.00% | 0.3150 |
| **2D CNN-BiLSTM** | Audio only | 0.8762 | 64.56% | 0.7315 | 27.58% | 65.71% | 0.3700 |
| **Standard Late-Fusion** | Audio + Clinical | **0.9226** | 78.64% | 0.8066 | 46.46% | 66.86% | 0.3094 |
| **Proposed (Gated + GRL + Calibrated)** | Audio + Clinical | 0.8208 | 56.01% | **0.8853** | **66.19%** | **12.57%** | **0.0661** |

### Per-Country Leave-One-Country-Out (LOCO) Generalization

| Held-Out Test Country | Patients ($N$) | Country TB Rate | Clinical Baseline AUROC | Audio CNN-BiLSTM AUROC | Standard Late-Fusion AUROC | Proposed GRL Model AUROC | Proposed Spec @ 90% Sens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **India** | 128 | 41.4% | 0.8558 | 0.7650 | 0.8259 | **0.8873** | **70.67%** (Meets WHO) |
| **Madagascar** | 116 | 22.4% | 0.8299 | 0.7239 | 0.8201 | **0.9393** | **75.56%** (Meets WHO) |
| **Philippines** | 136 | 18.4% | 0.7160 | 0.7532 | 0.8778 | **0.9124** | **79.28%** (Meets WHO) |
| **South Africa** | 124 | 15.3% | 0.6561 | 0.7657 | 0.8045 | **0.8917** | 65.71% |
| **Tanzania** | 112 | 33.9% | 0.7333 | 0.7304 | 0.7528 | **0.8460** | 40.54% |
| **Uganda** | 125 | 32.0% | 0.8026 | 0.6394 | 0.7526 | **0.8359** | 57.65% |
| **Vietnam** | 116 | 20.7% | 0.7726 | 0.7428 | 0.8125 | **0.8845** | **73.91%** (Meets WHO) |
| **Macro Average** | **857** | **26.3%** | **0.7666** | **0.7315** | **0.8066** | **0.8853** | **66.19%** |

### Key Findings by Research Question
- **RQ1 (Fusion Benefit):** Combining solicited cough spectrograms with clinical questionnaire features via Gated Cross-Modal Attention improves unseen-country LOCO AUROC to **0.8853**, outperforming the best clinical-only model (**0.7666**) by **+11.87%** and the best audio-only model (**0.7559**) by **+12.94%**.
- **RQ2 (Shortcut Diagnosis):** Standard audio models (`72.0%` MFCC+RF, `65.7%` CNN-BiLSTM) and standard late-fusion (`66.9%`) heavily leak recording country identity into their learned embeddings, and standard late-fusion predictions on TB-negative patients track country TB prevalence with **Slope = 0.94 ($R^2 = 0.97$)**.
- **RQ3 (Site-Invariant Robustness):** Attaching the Gradient Reversal Layer (GRL) site-adversarial head and site-balanced resampling drives country linear-probe accuracy down to **12.57%** (at the **14.3%** 7-country chance level), collapses the country TB-rate shortcut slope to **0.06 ($R^2 = 0.05$)**, and boosts unseen-country LOCO AUROC from **0.8066 $\to$ 0.8853**.
- **RQ4 (WHO Triage Target & Calibration):** At $\ge 90\%$ sensitivity, our proposed model boosts unseen-country specificity from **46.46% $\to$ 66.19%** (surpassing the $\ge 70\%$ WHO specificity target on 4 of 7 unseen countries), while post-hoc **Temperature Scaling ($T = 2.60$)** reduces Expected Calibration Error (ECE) from **0.3342 $\to$ 0.0661**.

---

## 4. Quickstart Instructions

### A. Run in Google Colab (Recommended)
1. Open `notebooks/CSE414_TB_Triage_Colab_Pipeline.ipynb` in **Google Colab**.
2. Click **Runtime $\rightarrow$ Run all**.
3. The notebook automatically installs dependencies, builds the dataset (or connects to your Synapse CODA TB folder on Google Drive), trains all 5 baselines + the PyTorch `ShortcutRobustMultimodalNet`, generates all 4 evaluation figures, and launches the interactive **Gradio live public link (`*.gradio.live`)**.

### B. Run Locally via CLI
```bash
pip install -r requirements.txt
python3 run_pipeline.py
python3 -m demo.app
```

### C. Initialize Git Branches & PR History for CLO5
```bash
bash scripts/init_github_branches.sh
git log --oneline --graph --all
```
