"""
numpy: Vectorized calculation of institutional High/Low nodes.
 pandas: Management of time-series OHLCV data for 20, 40, and 60-day lookbacks.
 dataclasses: For structured telemetry of the gravitational shift.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class EquilibriumTelemetry:
    current_price: float
    equilibrium_level: float
    zone: str              # PREMIUM or DISCOUNT
    cross_event: bool      # True if price crossed EQ in the last candle
    cross_direction: str   # BULLISH_CROSS or BEARISH_CROSS
    confidence: float      # Range coherence score

class EquilibriumCrossDetector:
    """
    IPDA Layer 1 Component: Identifies shifts across the 50% Mean.
    Gravity-centered detection based on 20, 40, and 60-day structural nodes.
    """
    def __init__(self, lookbacks=[7-9]):
        self.lookbacks = lookbacks
        self.prev_price = None
        self.equilibrium = None

    def _calculate_equilibrium(self, df: pd.DataFrame) -> float:
        """Calculates the absolute Mean of synchronized IPDA lookbacks."""
        levels = []
        for lb in self.lookbacks:
            h = df['high'].tail(lb).max()
            l = df['low'].tail(lb).min()
            levels.append((h + l) / 2)
        
        # Absolute Equilibrium: Mean of the structural nodes [6]
        return float(np.mean(levels))

    def process_tick(self, df: pd.DataFrame) -> EquilibriumTelemetry:
        """
        Master Logic: Detects the displacement across the gravitational center.
        """
        current_price = df['close'].iloc[-1]
        self.equilibrium = self._calculate_equilibrium(df)
        
        # Determine Current Zone [3]
        zone = "PREMIUM" if current_price > self.equilibrium else "DISCOUNT"
        
        # Detect Cross Event
        cross_event = False
        cross_direction = "NONE"
        
        if self.prev_price is not None:
            # Bullish Cross: Below -> Above
            if self.prev_price <= self.equilibrium and current_price > self.equilibrium:
                cross_event = True
                cross_direction = "BULLISH_CROSS"
            # Bearish Cross: Above -> Below
            elif self.prev_price >= self.equilibrium and current_price < self.equilibrium:
                cross_event = True
                cross_direction = "BEARISH_CROSS"

        # Coherence: 1 - variance of range widths [10]
        widths = [df['high'].tail(lb).max() - df['low'].tail(lb).min() for lb in self.lookbacks]
        coherence = 1 - (np.std(widths) / (np.mean(widths) + 1e-9))

        self.prev_price = current_price

        return EquilibriumTelemetry(
            current_price=current_price,
            equilibrium_level=self.equilibrium,
            zone=zone,
            cross_event=cross_event,
            cross_direction=cross_direction,
            confidence=round(coherence, 4)
        )
