from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import librosa
import pyloudnorm as pyln
from math import gcd

'''
As a note here, this is just hte mathematical computations, stuff like danceability
will need a model or other solution
'''

#Should all already be mono just in case though
def _to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    if y.shape[0] < y.shape[-1]:
        # likely (ch, n)
        return np.mean(y, axis=0)
    return np.mean(y, axis=1)


def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    # polyphase resampling

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(y, up, down).astype(np.float32, copy=False)


@dataclass
class AudioFeatureExtractor:
    y: np.ndarray
    sr: int

    def __init__(self, path, target_sr = None, max_duration_s: Optional[float] = None, trim_silence = True):
        y, sr = sf.read(path, always_2d=False)
        y = _to_mono(np.asarray(y))

        if target_sr is not None and target_sr != sr:
            y = _resample(y, sr, target_sr)
            sr = target_sr

        if max_duration_s is not None:
            n = int(max_duration_s * sr)
            y = y[:n]

        if trim_silence:
            # librosa trim expects float, mono
            y, _ = librosa.effects.trim(y, top_db=40)

        self.y = y
        self.sr = sr
        self._stft_magnitude = {}

    # ---------- Basic helpers ----------
    def duration_ms(self) -> float:
        return float(len(self.y) / self.sr * 1000.0)

    def rms(self, frame_length: int = 2048, hop_length: int = 512) -> float:
        r = librosa.feature.rms(y=self.y, frame_length=frame_length, hop_length=hop_length)[0]
        return float(np.mean(r))

    def rms_dbfs(self) -> float:
        r = np.sqrt(np.mean(self.y ** 2))
        r = max(r, 1e-12)
        return float(20.0 * np.log10(r))  # relative to full-scale if y is normalized to [-1,1]

    def peak(self) -> float:
        return float(np.max(np.abs(self.y)))

    def crest_factor_db(self) -> float:
        r = np.sqrt(np.mean(self.y ** 2))
        r = max(r, 1e-12)
        p = max(self.peak(), 1e-12)
        return float(20.0 * np.log10(p / r))

    # ---------- Spotify-ish proxies ----------
    def energy_proxy_0_1(self) -> float:
        """
        A simple proxy for "energy": squash RMS into [0,1].
        This won't match Spotify's energy but is stable and useful for clustering.
        """
        r = self.rms()
        # squash: typical music RMS might fall ~[0.02, 0.2] depending on normalization
        # adjust if your pipeline differs
        x = (r - 0.02) / (0.20 - 0.02)
        return float(np.clip(x, 0.0, 1.0))

    def loudness_lufs(self) -> Optional[float]:
        """
        Integrated loudness in LUFS if pyloudnorm is installed.
        """
        if pyln is None:
            return None
        meter = pyln.Meter(self.sr)
        # pyloudnorm expects float in [-1,1]
        return float(meter.integrated_loudness(self.y))

    def tempo_bpm(self) -> float:
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sr, aggregate=np.median)
        return float(tempo[0])

    def key_and_mode(self) -> Tuple[int, int, float]:
        """
        Returns (key, mode, confidence)
        - key: 0=C, 1=C#, ... 11=B
        - mode: 1=major, 0=minor
        confidence is template correlation gap.
        """
        chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_mean = chroma_mean / (np.sum(chroma_mean) + 1e-12)

        # Krumhansl major/minor templates (normalized)
        major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        major = major / major.sum()
        minor = minor / minor.sum()

        def best_corr(template: np.ndarray) -> Tuple[int, float]:
            corrs = []
            for k in range(12):
                corrs.append(np.dot(chroma_mean, np.roll(template, k)))
            k_best = int(np.argmax(corrs))
            return k_best, float(np.max(corrs))

        kM, cM = best_corr(major)
        km, cm = best_corr(minor)

        if cM >= cm:
            key, mode = kM, 1
            conf = cM - cm
        else:
            key, mode = km, 0
            conf = cm - cM

        return key, mode, float(conf)

    def time_signature_guess(self) -> Tuple[int, float]:
        """
        Very rough heuristic: guess 3 vs 4 based on periodic accents across beats.
        Returns (time_signature, confidence).
        """
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        if beats is None or len(beats) < 8:
            return 4, 0.0  # default-ish

        # Beat-synchronous onset energy (accent per beat)
        beat_env = librosa.util.sync(onset_env.reshape(1, -1), beats, aggregate=np.mean)[0]
        beat_env = beat_env - np.mean(beat_env)
        if np.std(beat_env) < 1e-6:
            return 4, 0.0

        # Autocorrelation over beats; check lags 3 and 4
        ac = np.correlate(beat_env, beat_env, mode="full")
        ac = ac[len(ac)//2:]  # non-negative lags
        # guard
        if len(ac) <= 5:
            return 4, 0.0

        lag3 = ac[3] if len(ac) > 3 else 0.0
        lag4 = ac[4] if len(ac) > 4 else 0.0

        if lag3 > lag4:
            conf = (lag3 - lag4) / (abs(lag3) + abs(lag4) + 1e-12)
            return 3, float(conf)
        else:
            conf = (lag4 - lag3) / (abs(lag3) + abs(lag4) + 1e-12)
            return 4, float(conf)

    # ---------- Extra DSP features (great for clustering) ----------
    def spectral_summary(self) -> Dict[str, float]:
        centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr, roll_percent=0.85)[0]
        flatness = librosa.feature.spectral_flatness(y=self.y)[0]
        zcr = librosa.feature.zero_crossing_rate(y=self.y)[0]
        return {
            "spectral_centroid_mean": float(np.mean(centroid)),
            "spectral_bandwidth_mean": float(np.mean(bandwidth)),
            "spectral_rolloff85_mean": float(np.mean(rolloff)),
            "spectral_flatness_mean": float(np.mean(flatness)),
            "zcr_mean": float(np.mean(zcr)),
        }

    def harmonic_percussive_ratio(self) -> float:
        y_harm, y_perc = librosa.effects.hpss(self.y)
        eh = float(np.mean(y_harm ** 2))
        ep = float(np.mean(y_perc ** 2))
        return float(eh / (eh + ep + 1e-12))

    def compute_all(self) -> Dict[str, Any]:
        key, mode, key_conf = self.key_and_mode()
        ts, ts_conf = self.time_signature_guess()

        feats: Dict[str, Any] = {
            "duration_ms": self.duration_ms(),
            "tempo": self.tempo_bpm(),
            "loudness_lufs": self.loudness_lufs(),   # None if pyloudnorm not installed
            "loudness_rms_dbfs": self.rms_dbfs(),   # always available
            "energy_proxy": self.energy_proxy_0_1(),
            "key": key,
            "mode": mode,
            "key_confidence": key_conf,
            "time_signature_guess": ts,
            "time_signature_confidence": ts_conf,
            "crest_factor_db": self.crest_factor_db(),
            "harmonic_ratio": self.harmonic_percussive_ratio(),
        }
        feats.update(self.spectral_summary())
        return feats


if __name__ == "__main__":
    wav_path = r"/Users/Riley/Downloads/archive-3/audio_waveforms/Lady Gaga, Bruno Mars - Die With A Smile (Audio).wav"
    extractor = AudioFeatureExtractor(wav_path, target_sr=48000, max_duration_s=30)

    features = extractor.compute_all()
    for k, v in features.items():
        print(f"{k:28s} : {v}")