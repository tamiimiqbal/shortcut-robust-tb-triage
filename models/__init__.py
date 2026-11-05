from .adversarial import GradientReversalLayer, SiteAdversarialHead, compute_dann_lambda
from .audio_encoder import CoughCNN2D, PatientAudioEncoder, PretrainedAudioEncoderWrapper
from .clinical_encoder import ClinicalMLPEncoder
from .fusion import LateFusionModule, GatedCrossModalAttentionFusion
from .losses import FocalLossWithLogits
from .calibration import TemperatureScaler
from .multimodal_net import ShortcutRobustMultimodalNet
from .baselines import (
    ClinicalLogisticRegression,
    ClinicalRandomForest,
    ClinicalXGBoost,
    AudioMFCCRandomForest,
)

__all__ = [
    "GradientReversalLayer",
    "SiteAdversarialHead",
    "compute_dann_lambda",
    "CoughCNN2D",
    "PatientAudioEncoder",
    "PretrainedAudioEncoderWrapper",
    "ClinicalMLPEncoder",
    "LateFusionModule",
    "GatedCrossModalAttentionFusion",
    "FocalLossWithLogits",
    "TemperatureScaler",
    "ShortcutRobustMultimodalNet",
    "ClinicalLogisticRegression",
    "ClinicalRandomForest",
    "ClinicalXGBoost",
    "AudioMFCCRandomForest",
]
