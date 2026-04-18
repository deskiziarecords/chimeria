"""
numpy: Vectorized calculation of institutional range midpoints.
    pandas: Management of time-series OHLCV history for lookback synchronization.
    dataclasses: For structured bias telemetry and state tracking.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class BiasState:
    bias: str            # BULLISH, BEARISH, NEUTRAL
    equilibrium: float    # Absolute IPDA 50% Mean
    distance: float       # Distance from Equilibrium in pips
    zone: str             # PREMIUM or DISCOUNT
    coherence: float      # Range alignment score (0.0 - 1.0)
    is_valid: bool        # Structural integrity check

class BiasDetector:
    """
    IPDA Layer 1 Component: Structural Orientation Engine.
    Identifies institutional bias relative to the 20/40/60-day equilibrium.
    """
    def __init__(self, lookbacks=[20, 40, 60], neutral_threshold_atr: float = 0.15):
        self.lookbacks = lookbacks
        self.threshold_mult = neutral_threshold_atr # Neutral if within 0.15 * ATR

    def calculate_absolute_equilibrium(self, df: pd.DataFrame) -> float:
        """Calculates the mean of the 20, 40, and 60-period midpoints [1, 8]."""
        midpoints = []
        for lb in self.lookbacks:
            h = df['high'].tail(lb).max()
            l = df['low'].tail(lb).min()
            midpoints.append((h + l) / 2)
        
        # Absolute Equilibrium: The Market's 'Fair Price' [3]
        return float(np.mean(midpoints))

    def detect_bias(self, df: pd.DataFrame) -> BiasState:
        """
        Master Logic: Maps price to structural bias.
        Bullish: Price > Equilibrium (Momentum delivery)
        Bearish: Price < Equilibrium (Momentum delivery)
        Neutral: Price within proximity of Equilibrium (Stasis/Accumulation)
        """
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df else (df['high'] - df['low']).mean()
        
        eq = self.calculate_absolute_equilibrium(df)
        diff = current_price - eq
        
        # Proximity logic for NEUTRAL state (Accumulation Phase) [9]
        neutral_bound = atr * self.threshold_mult
        
        if abs(diff) < neutral_bound:
            bias = "NEUTRAL"
        elif diff > 0:
            bias = "BULLISH"
        else:
            bias = "BEARISH"

        # Structural Metadata
        zone = "PREMIUM" if current_price > eq else "DISCOUNT"
        
        # Range Coherence: 1 - variance of range widths [10]
        widths = [df['high'].tail(lb).max() - df['low'].tail(lb).min() for lb in self.lookbacks]
        coherence = 1 - (np.std(widths) / (np.mean(widths) + 1e-9))

        return BiasState(
            bias=bias,
            equilibrium=round(eq, 5),
            distance=round(abs(diff), 5),
            zone=zone,
            coherence=round(coherence, 4),
            is_valid=coherence > 0.4 # Invalid if ranges are non-coherent [10]
        )

# --- EXECUTION ---
# detector = BiasDetector()
# current_bias = detector.detect_bias(ohlcv_df)

