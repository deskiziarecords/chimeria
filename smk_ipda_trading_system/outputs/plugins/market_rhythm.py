"""
plugins/market_rhythm.py
L3-ext — MarketRhythmEngine
MIR-to-OHLCV: spectrogram, beat detection, λ3 harmony, Shazam fingerprint.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from plugins import SMKPlugin


class MarketRhythmPlugin(SMKPlugin):
    name            = "MarketRhythm"
    layer           = "L3-ext"
    sensor_id       = "p01"
    requires_warmup = 64

    def __init__(self, sample_rate: int = 128, lookback: int = 64,
                 beat_threshold: float = 1.5, spectral_peak_threshold: float = 2.0):
        super().__init__()
        self.fs                      = sample_rate
        self.lookback                = lookback
        self.beat_threshold          = beat_threshold
        self.spectral_peak_threshold = spectral_peak_threshold
        # Rolling fingerprint history for pattern matching
        self._fingerprint_history: list[str] = []

    # ── helpers ───────────────────────────────────────────────────────────────
    def _to_waveform(self, prices: np.ndarray) -> np.ndarray:
        diff = np.diff(prices)
        return diff / (np.std(diff) + 1e-9)

    def _detect_beats(self, high: np.ndarray, low: np.ndarray) -> dict:
        if len(high) < 3:
            return {"tempo": 0.0, "beat_count": 0, "beat_locations": []}
        swings_h = (high[1:-1] > high[:-2]) & (high[1:-1] > high[2:])
        swings_l = (low[1:-1]  < low[:-2])  & (low[1:-1]  < low[2:])
        n_beats   = int(np.sum(swings_h) + np.sum(swings_l))
        tempo     = n_beats / (len(high) / max(self.fs, 1))
        locs      = np.where(swings_h | swings_l)[0] + 1
        return {"tempo": float(tempo), "beat_count": n_beats,
                "beat_locations": locs.tolist()[-10:]}  # last 10 only

    def _check_harmony(self, l1_bias_phi: float, waveform: np.ndarray) -> tuple[bool, float]:
        fft_data = np.fft.rfft(waveform)
        if len(fft_data) < 2:
            return True, 0.0
        dominant_idx = int(np.argmax(np.abs(fft_data[1:]))) + 1
        phi_l5       = float(np.angle(fft_data[dominant_idx]))
        phase_diff   = abs(phi_l5 - l1_bias_phi)
        return phase_diff < (np.pi / 2), float(round(phase_diff, 4))

    def _fingerprint(self, waveform: np.ndarray) -> str:
        if len(waveform) < self.lookback:
            return "INSUFFICIENT"
        try:
            _, _, spec = scipy_signal.spectrogram(
                waveform[-self.lookback:], fs=self.fs, nperseg=min(32, len(waveform)))
            mean_s = float(np.mean(spec))
            std_s  = float(np.std(spec))
            peaks  = spec > (mean_s + self.spectral_peak_threshold * std_s)
            return str(hash(peaks.tobytes()))[:12]
        except Exception:
            return "ERR"

    def _fingerprint_repeat(self, fp: str) -> bool:
        """True if this fingerprint appeared in recent history — pattern repeat."""
        return fp in self._fingerprint_history[-50:]

    # ── main ──────────────────────────────────────────────────────────────────
    def update(self, bar: dict, df: pd.DataFrame, smk: dict) -> dict:
        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values

        waveform  = self._to_waveform(closes)
        beats     = self._detect_beats(highs, lows)

        # L1 bias phi: convert BULLISH/BEARISH to radian proxy
        bias      = smk.get("bias", {}).get("bias", "NEUTRAL")
        l1_phi    = 0.5 if bias == "BULLISH" else -0.5 if bias == "BEARISH" else 0.0

        in_harmony, phase_diff = self._check_harmony(l1_phi, waveform)
        fp = self._fingerprint(waveform)

        repeat = self._fingerprint_repeat(fp)
        self._fingerprint_history.append(fp)
        if len(self._fingerprint_history) > 200:
            self._fingerprint_history = self._fingerprint_history[-200:]

        status = "IN_HARMONY" if in_harmony else "DISSONANT:λ3_VETO"
        score  = float(min(1.0, beats["tempo"] / 10.0))

        return {
            "tempo_bpm":    float(round(beats["tempo"], 3)),
            "beat_count":   int(beats["beat_count"]),
            "is_harmonic":  bool(in_harmony),
            "phase_diff":   float(phase_diff),
            "fingerprint":  str(fp),
            "pattern_repeat": bool(repeat),
            "status":       str(status),
            "active":       bool(not in_harmony),   # active = VETO condition
            "score":        float(score),
        }
