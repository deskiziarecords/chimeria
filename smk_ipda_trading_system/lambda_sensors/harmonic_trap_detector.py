"""
numpy: For vectorized waveform manipulation and FFT processing.
    scipy.signal: For spectrogram generation and Butterworth filtering to isolate market noise.
    dataclasses: For structured spectral telemetry and trap classification.
 """   
import numpy as np
import scipy.signal as signal
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class HarmonicTelemetry:
    phase_difference: float
    is_inverted: bool           # λ3 Trigger Status
    dominant_frequency: float
    trap_type: str              # PhaseInversion, FrequencyDoubling, etc.
    status: str

class HarmonicTrapDetector:
    """
    Sovereign Market Kernel: λ3 Spectral Inversion Engine.
    Identifies Harmonic Traps and Reverse Periods via FFT-based phase analysis.
    """
    def __init__(self, threshold: float = np.pi / 2, lookback: int = 64):
        self.threshold = threshold  # 90-degree "Death Cross" of Phase
        self.lookback = lookback    # Typical FFT rolling window [8]
        self.trap_types = {
            "PHASE_INVERSION": "Shift > 90°",
            "FREQUENCY_DOUBLING": "2x Frequency component",
            "SPECTRAL_FOLD": "Aliasing pattern detected"
        }

    def _extract_dominant_phase(self, signal_data: np.ndarray) -> Tuple[float, float]:
        """Extracts the phase and value of the dominant frequency component."""
        fft_vals = np.fft.rfft(signal_data)
        amplitudes = np.abs(fft_vals)
        
        # Identify peak frequency (ignoring DC component at index 0)
        dom_freq_idx = np.argmax(amplitudes[1:]) + 1
        phase = np.angle(fft_vals[dom_freq_idx])
        return phase, float(dom_freq_idx)

    def detect_trap(self, predicted_prices: np.ndarray, actual_prices: np.ndarray) -> HarmonicTelemetry:
        """
        Master Logic: Compares predicted vs actual spectral phases.
        Trigger: abs(phi_pred - phi_act) > pi/2 [2, 9].
        """
        if len(actual_prices) < self.lookback:
            return HarmonicTelemetry(0, False, 0, "NONE", "INSUFFICIENT_DATA")

        # 1. Spectral Extraction
        phi_pred, freq_pred = self._extract_dominant_phase(predicted_prices[-self.lookback:])
        phi_act, freq_act = self._extract_dominant_phase(actual_prices[-self.lookback:])

        # 2. Phase Difference Calculation
        phase_diff = np.abs(phi_pred - phi_act)
        
        # 3. λ3 Veto Condition
        is_inverted = phase_diff > self.threshold
        
        # 4. Trap Classification [10]
        trap_type = "NONE"
        if is_inverted:
            trap_type = "PHASE_INVERSION"
        elif freq_act > freq_pred * 1.8:
            trap_type = "FREQUENCY_DOUBLING"

        status = "DISSONANT: λ3 VETO" if is_inverted else "IN_HARMONY"
        
        return HarmonicTelemetry(
            phase_difference=float(phase_diff),
            is_inverted=is_inverted,
            dominant_frequency=freq_act,
            trap_type=trap_type,
            status=status
        )

