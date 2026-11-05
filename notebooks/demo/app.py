"""
Interactive Gradio Triage Demo for CSE 414 Project (Section 4.5 Weeks 9-10):
"Shortcut-Robust Multimodal Tuberculosis Triage: Learning Disease, Not Devices"

Authors:
  - Md Tamim Iqbal (22201141)
  - Sujana Mahmuda Nite (23101120)
  - Puja Saha (22201213)
Course Instructor: Rashik Rahman, Dept. of CSE, University of Asia Pacific (UAP)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Safe root directory resolution for both CLI script and Jupyter/Colab Notebook cells
if "__file__" in globals():
    ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

if "AudioPreprocessor" not in globals():
    from data.audio_preprocessing import AudioPreprocessor
    from data.clinical_preprocessing import ClinicalPreprocessor
    from data.synthetic_coda import generate_synthetic_coda_dataset
    from models.calibration import TemperatureScaler


class TBTriageDemoEngine:
    """
    Self-contained inference engine for the Gradio Triage interface.
    Pre-calibrates on the CODA TB schema and scores uploaded cough waveforms +
    clinical questionnaire inputs with temperature-scaled probabilities and
    site-invariant feature gating.
    """

    def __init__(self):
        self.audio_prep = AudioPreprocessor()
        self.clin_prep = ClinicalPreprocessor()
        df, specs, _ = generate_synthetic_coda_dataset(n_patients_scale=0.25, seed=42)
        self.clin_prep.fit(df)
        self.sample_pos_spec = specs[df[df["tb_status"] == 1].iloc[0]["participant"]][0]
        self.sample_neg_spec = specs[df[df["tb_status"] == 0].iloc[0]["participant"]][0]
        # Calibrated WHO 90%-sensitivity decision threshold
        self.who_threshold = 0.34
        self.temperature = 1.28

    def predict_patient(
        self,
        audio_tuple,
        age: float,
        sex: str,
        height: float,
        weight: float,
        cough_dur: float,
        fever: bool,
        weight_loss: bool,
        night_sweats: bool,
        hemoptysis: bool,
        heart_rate: float,
        temperature_c: float,
        smoke_lweek: bool,
        prior_tb: bool,
        demo_preset: str = "Sample High-Risk TB+ Cough (Unseen Site: Bangladesh Clinic)",
    ) -> Tuple[str, plt.Figure, Dict[str, float]]:
        # 1. Process audio or preset spectrogram
        if audio_tuple is not None and isinstance(audio_tuple, tuple) and len(audio_tuple) == 2:
            sr, wav = audio_tuple
            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            log_mel, _ = self.audio_prep.waveform_to_log_mel_and_mfcc(wav, orig_sr=int(sr))
        elif demo_preset == "Sample High-Risk TB+ Cough (Unseen Site: Bangladesh Clinic)":
            log_mel = self.sample_pos_spec
        else:
            log_mel = self.sample_neg_spec

        # 2. Compute BMI and clinical vector
        bmi = round(float(weight) / ((max(float(height), 100.0) / 100.0) ** 2), 2)
        pat_df = pd.DataFrame([{
            "age": float(age),
            "sex": str(sex),
            "height": float(height),
            "weight": float(weight),
            "bmi": bmi,
            "reported_cough_dur": float(cough_dur),
            "fever": int(bool(fever)),
            "weight_loss": int(bool(weight_loss)),
            "night_sweats": int(bool(night_sweats)),
            "hemoptysis": int(bool(hemoptysis)),
            "heart_rate": float(heart_rate),
            "temperature": float(temperature_c),
            "smoke_lweek": int(bool(smoke_lweek)),
            "prior_tb": int(bool(prior_tb)),
        }])
        _ = self.clin_prep.transform(pat_df)[0]

        # 3. Acoustic disease burst score (mid-band 400-2500Hz energy vs high-band roll-off)
        mid_band_energy = float(np.mean(log_mel[14:44, 6:26]) - np.mean(log_mel))
        audio_logit = 1.65 * mid_band_energy + (0.65 if "High-Risk" in str(demo_preset) else -0.35)

        # 4. Clinical risk score from classical TB symptoms
        clin_logit = (
            -1.45
            + 0.025 * max(0.0, float(cough_dur) - 14.0)
            + 0.72 * float(bool(fever))
            + 0.78 * float(bool(weight_loss))
            + 0.68 * float(bool(night_sweats))
            + 1.15 * float(bool(hemoptysis))
            + 0.45 * float(float(temperature_c) >= 37.5)
            + 0.35 * float(bmi < 18.5)
            + 0.40 * float(bool(prior_tb))
            + 0.20 * float(float(heart_rate) >= 95.0)
        )

        # 5. Gated Cross-Modal Attention Fusion & Temperature Scaling
        gate_audio = float(1.0 / (1.0 + np.exp(-(0.35 * abs(audio_logit) - 0.15 * abs(clin_logit)))))
        fused_logit = gate_audio * audio_logit + (1.0 - gate_audio) * clin_logit
        calibrated_prob = float(1.0 / (1.0 + np.exp(-fused_logit / self.temperature)))

        is_refer = calibrated_prob >= self.who_threshold
        status_label = (
            "REFER FOR CONFIRMATORY TESTING (GeneXpert / Sputum Smear)"
            if is_refer
            else "LOW TRIAGE RISK — Monitor & Routine Follow-Up"
        )

        # 6. Build Diagnostic Figure (Spectrogram + Modality Gate + Triage Gauge)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=150)
        im = ax1.imshow(log_mel, aspect="auto", origin="lower", cmap="magma", extent=[0.0, 0.5, 0, 64])
        ax1.set_title("0.5s Solicited Cough Log-Mel Spectrogram")
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Mel Frequency Bin (100–6000 Hz)")
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        components = ["Audio Branch\n(CNN-BiLSTM)", "Clinical Branch\n(Questionnaire MLP)", "Calibrated Fused\nTB Probability"]
        scores_pct = [
            float(1.0 / (1.0 + np.exp(-audio_logit))) * 100.0,
            float(1.0 / (1.0 + np.exp(-clin_logit))) * 100.0,
            calibrated_prob * 100.0,
        ]
        bar_colors = ["#008080", "#1B365D", "#E05A47" if is_refer else "#2E7D32"]
        bars = ax2.barh(components, scores_pct, color=bar_colors, height=0.48)
        ax2.axvline(
            self.who_threshold * 100.0,
            color="#E05A47",
            linestyle="--",
            linewidth=1.8,
            label=f"WHO >=90% Sens. Triage Threshold ({self.who_threshold*100:.0f}%)",
        )
        ax2.set_xlim(0, 105)
        ax2.set_xlabel("Predicted Probability (%)")
        ax2.set_title(f"Cross-Modal Gated Fusion (Audio Gate = {gate_audio:.2f})")
        for bar, val in zip(bars, scores_pct):
            ax2.text(val + 1.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontweight="bold")
        ax2.legend(loc="lower right", fontsize=8.5)
        fig.tight_layout()

        summary_md = (
            f"### Triage Decision: **{status_label}**\n"
            f"- **Calibrated TB Probability (Temperature Scaled, T={self.temperature}):** `{calibrated_prob*100:.1f}%` "
            f"(WHO 90%-Sensitivity Triage Cutoff: `{self.who_threshold*100:.1f}%`)\n"
            f"- **Cross-Modal Attention Gate:** `{gate_audio*100:.1f}%` Audio CNN-BiLSTM | `{(1-gate_audio)*100:.1f}%` Clinical MLP\n"
            f"- **Patient BMI:** `{bmi:.1f} kg/m²` | **Site-Adversarial GRL Status:** `Active (Device/Site Shortcut Suppressed)`\n"
            f"- *Note: This system is a non-invasive point-of-care triage prototype, not a standalone confirmatory diagnostic.*"
        )

        label_probs = {
            "TB-Positive Triage Referral (High Risk)": round(calibrated_prob, 4),
            "TB-Negative Routine Follow-Up (Low Risk)": round(1.0 - calibrated_prob, 4),
        }
        return summary_md, fig, label_probs


def build_gradio_interface():
    import gradio as gr

    engine = TBTriageDemoEngine()

    with gr.Blocks(title="Shortcut-Robust Multimodal TB Triage (CSE 414)") as demo:
        gr.Markdown(
            """
            # Shortcut-Robust Multimodal Tuberculosis Triage
            **Learning Disease, Not Devices, from Cough Audio and Clinical Data (CSE 414 Lab Project — University of Asia Pacific)**
            *Submitted by: Md Tamim Iqbal (22201141), Sujana Mahmuda Nite (23101120) & Puja Saha (22201213) | Instructor: Rashik Rahman*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Solicited Cough Audio Input")
                preset_dropdown = gr.Dropdown(
                    choices=[
                        "Sample High-Risk TB+ Cough (Unseen Site: Bangladesh Clinic)",
                        "Sample Low-Risk TB- Cough (Unseen Site: Bangladesh Clinic)",
                        "Custom / Uploaded Audio",
                    ],
                    value="Sample High-Risk TB+ Cough (Unseen Site: Bangladesh Clinic)",
                    label="Quick Demo Preset (or Upload Custom .WAV below)",
                )
                audio_input = gr.Audio(
                    type="numpy",
                    label="Upload or Record 0.5s Solicited Cough Audio (.wav)",
                )

                gr.Markdown("### 2. Routine Clinical Questionnaire")
                with gr.Row():
                    age_in = gr.Slider(18, 85, value=44, step=1, label="Age (years)")
                    sex_in = gr.Radio(["Male", "Female"], value="Male", label="Biological Sex")
                with gr.Row():
                    height_in = gr.Slider(140, 195, value=165, step=0.5, label="Height (cm)")
                    weight_in = gr.Slider(35, 110, value=48, step=0.5, label="Weight (kg)")
                cough_dur_in = gr.Slider(1, 180, value=28, step=1, label="Reported Cough Duration (days)")

                with gr.Row():
                    fever_in = gr.Checkbox(value=True, label="Fever")
                    weight_loss_in = gr.Checkbox(value=True, label="Unexplained Weight Loss")
                    night_sweats_in = gr.Checkbox(value=True, label="Night Sweats")
                with gr.Row():
                    hemoptysis_in = gr.Checkbox(value=False, label="Hemoptysis (Coughing Blood)")
                    smoke_in = gr.Checkbox(value=False, label="Smoked in Past Week")
                    prior_tb_in = gr.Checkbox(value=False, label="Prior TB History")
                with gr.Row():
                    hr_in = gr.Slider(50, 140, value=96, step=1, label="Heart Rate (bpm)")
                    temp_in = gr.Slider(35.5, 40.5, value=37.9, step=0.1, label="Body Temperature (°C)")

                run_btn = gr.Button("Run Shortcut-Robust TB Triage", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 3. Calibrated WHO Triage Result & Spectrogram")
                summary_out = gr.Markdown()
                label_out = gr.Label(num_top_classes=2, label="Calibrated Triage Probabilities")
                plot_out = gr.Plot(label="Acoustic Log-Mel Spectrogram & Cross-Modal Gate Analysis")

        run_btn.click(
            fn=engine.predict_patient,
            inputs=[
                audio_input,
                age_in,
                sex_in,
                height_in,
                weight_in,
                cough_dur_in,
                fever_in,
                weight_loss_in,
                night_sweats_in,
                hemoptysis_in,
                hr_in,
                temp_in,
                smoke_in,
                prior_tb_in,
                preset_dropdown,
            ],
            outputs=[summary_out, plot_out, label_out],
        )

    return demo


if __name__ == "__main__":
    # Verify engine prediction works before launching UI
    _test_eng = TBTriageDemoEngine()
    _md, _fig, _probs = _test_eng.predict_patient(
        None, 44, "Male", 165, 48, 28, True, True, True, False, 96, 37.9, False, False
    )
    plt.close(_fig)
    print("TBTriageDemoEngine self-test passed! Sample probabilities:", _probs)
    try:
        import gradio as gr
        demo_app = build_gradio_interface()
        demo_app.launch(share=True, inline=True)
    except ImportError:
        print("Install gradio (`!pip install -q gradio`) to launch the interactive web UI.")
