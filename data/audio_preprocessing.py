"""
Audio Preprocessing Pipeline for Solicited Cough Recordings (Section 4.1).

Handles:
1. Resampling raw audio to 16,000 Hz
2. 4th-order Butterworth bandpass filtering (100 Hz - 6,000 Hz) to remove DC offset & high-freq hiss
3. Extracting/padding 0.5s cough windows (8,000 samples)
4. Computing 64-bin Log-Mel Spectrograms and 40-dim MFCC features
"""

from typing import Tuple
import numpy as np
from scipy import signal
from scipy.fftpack import dct


class AudioPreprocessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        clip_duration_sec: float = 0.5,
        bandpass_low_hz: float = 100.0,
        bandpass_high_hz: float = 6000.0,
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mfcc: int = 40,
    ):
        self.sample_rate = sample_rate
        self.clip_samples = int(sample_rate * clip_duration_sec)
        self.bandpass_low_hz = bandpass_low_hz
        self.bandpass_high_hz = bandpass_high_hz
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.mel_basis = self._build_mel_filterbank()

    def _build_mel_filterbank(self) -> np.ndarray:
        """Create standard Slaney-style Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_min = hz_to_mel(self.bandpass_low_hz)
        mel_max = hz_to_mel(min(self.bandpass_high_hz, self.sample_rate / 2.0))
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        n_freqs = self.n_fft // 2 + 1
        weights = np.zeros((self.n_mels, n_freqs), dtype=np.float32)
        for m in range(1, self.n_mels + 1):
            f_left, f_center, f_right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
            for k in range(f_left, f_center):
                if f_center > f_left and 0 <= k < n_freqs:
                    weights[m - 1, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                if f_right > f_center and 0 <= k < n_freqs:
                    weights[m - 1, k] = (f_right - k) / (f_right - f_center)
        return weights

    def bandpass_filter(self, waveform: np.ndarray, orig_sr: int = 16000) -> np.ndarray:
        """Resample if needed and apply 4th-order Butterworth bandpass filter."""
        if orig_sr != self.sample_rate:
            num_target = int(round(len(waveform) * float(self.sample_rate) / orig_sr))
            waveform = signal.resample(waveform, num_target)
        nyq = 0.5 * self.sample_rate
        low = max(self.bandpass_low_hz / nyq, 1e-4)
        high = min(self.bandpass_high_hz / nyq, 0.99)
        sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
        filtered = signal.sosfiltfilt(sos, waveform).astype(np.float32)
        # Peak normalize per cough clip
        peak = np.max(np.abs(filtered)) + 1e-8
        return filtered / peak

    def pad_or_crop_05s(self, waveform: np.ndarray) -> np.ndarray:
        """Center-crop around peak cough energy or zero-pad to exactly 0.5s (8,000 samples)."""
        if len(waveform) == self.clip_samples:
            return waveform
        if len(waveform) < self.clip_samples:
            pad_total = self.clip_samples - len(waveform)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return np.pad(waveform, (pad_left, pad_right), mode="constant")
        # Center window around maximum energy frame
        energy = np.convolve(waveform ** 2, np.ones(512) / 512.0, mode="same")
        peak_idx = int(np.argmax(energy))
        half = self.clip_samples // 2
        start = max(0, min(len(waveform) - self.clip_samples, peak_idx - half))
        return waveform[start : start + self.clip_samples]

    def waveform_to_log_mel_and_mfcc(self, waveform: np.ndarray, orig_sr: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
        """Convert raw cough waveform into (64, 32) Log-Mel Spectrogram and (40,) MFCC vector."""
        clean = self.pad_or_crop_05s(self.bandpass_filter(waveform, orig_sr))
        _, _, zxx = signal.stft(
            clean,
            fs=self.sample_rate,
            window="hann",
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary="zeros",
            padded=True,
        )
        power_spec = np.abs(zxx) ** 2
        mel_spec = np.dot(self.mel_basis, power_spec)
        log_mel = np.log(mel_spec + 1e-6).astype(np.float32)
        # Ensure exact 32 time frames for 0.5s clip
        if log_mel.shape[1] < 32:
            log_mel = np.pad(log_mel, ((0, 0), (0, 32 - log_mel.shape[1])), mode="edge")
        else:
            log_mel = log_mel[:, :32]

        mfcc_frames = dct(log_mel, type=2, axis=0, norm="ortho")[: self.n_mfcc]
        mfcc_vec = mfcc_frames.mean(axis=1).astype(np.float32)
        return log_mel, mfcc_vec
