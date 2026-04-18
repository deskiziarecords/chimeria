import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class EntrapmentTelemetry:
    volatility_ratio: float
    is_entrapped: bool
    latent_energy_score: float
    time_in_stasis: int
    status: str

class VolatilityDecayDetector:
    """
    Sovereign Market Kernel: λ1 Phase Entrapment Sensor.
    Quantifies volatility exhaustion to predict institutional displacement.
    """
    def __init__(self, delta: float = 0.7, tau_max: int = 20):
        self.delta = delta      # Critical mass threshold (0.7δ) [2]
        self.tau_max = tau_max  # Stagnation persistence limit [5]
        self.stasis_timer = 0

    def calculate_price_variation(self, prices: np.ndarray) -> float:
        """Calculates the Price Variation Integral (Vt)."""
        # Sum of absolute price changes over the window [2]
        return np.sum(np.abs(np.diff(prices)))

    def detect_entrapment(self, df: pd.DataFrame) -> EntrapmentTelemetry:
        """
        Master Logic: Identifies mathematical exhaustion of volatility.
        Trigger: Vt / ATR20 < 0.7 [2, 6].
        """
        if len(df) < 20:
            return EntrapmentTelemetry(0, False, 0, 0, "INSUFFICIENT_DATA")

        close_prices = df['close'].values
        # Benchmarking against 20-period ATR
        atr20 = df['atr20'].iloc[-1]
        
        v_t = self.calculate_price_variation(close_prices)
        vol_ratio = v_t / (atr20 + 1e-9)
        
        # λ1 Condition: Volatility Ratio < 0.7 [2]
        is_entrapped = vol_ratio < self.delta
        
        if is_entrapped:
            self.stasis_timer += 1
        else:
            self.stasis_timer = 0

        # Model Latent Institutional Energy H(t) [3]
        # Potential Energy = 1/2 * k * |H(t)|^2
        latent_energy = 0.5 * (self.stasis_timer ** 2)
        
        status = "PHASE_ENTRAPMENT_ACTIVE" if is_entrapped else "NORMAL_DELIVERY"
        if self.stasis_timer > self.tau_max:
            status = "CRITICAL_MASS_EXPANSION_IMMINENT"

        return EntrapmentTelemetry(
            volatility_ratio=round(vol_ratio, 4),
            is_entrapped=is_entrapped,
            latent_energy_score=latent_energy,
            time_in_stasis=self.stasis_timer,
            status=status
        )
