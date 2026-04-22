"""
FIXED: lookbacks=[7-9] → [20, 40, 60]
Original bug: [7-9] evaluates to [-2], so only last 2 bars were used
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class EquilibriumTelemetry:
    current_price: float
    equilibrium_level: float
    zone: str
    cross_event: bool
    cross_direction: str
    confidence: float

class EquilibriumCrossDetector:
    def __init__(self, lookbacks=[20, 40, 60]):  # FIXED: was [7-9] → [-2]
        self.lookbacks = lookbacks
        self.prev_price = None
        self.equilibrium = None

    def _calculate_equilibrium(self, df: pd.DataFrame) -> float:
        levels = []
        for lb in self.lookbacks:
            h = df['high'].tail(lb).max()
            l = df['low'].tail(lb).min()
            levels.append((h + l) / 2)
        return float(np.mean(levels))

    def process_tick(self, df: pd.DataFrame) -> EquilibriumTelemetry:
        current_price = df['close'].iloc[-1]
        self.equilibrium = self._calculate_equilibrium(df)
        
        zone = "PREMIUM" if current_price > self.equilibrium else "DISCOUNT"
        
        cross_event = False
        cross_direction = "NONE"
        
        if self.prev_price is not None:
            if self.prev_price <= self.equilibrium and current_price > self.equilibrium:
                cross_event = True
                cross_direction = "BULLISH_CROSS"
            elif self.prev_price >= self.equilibrium and current_price < self.equilibrium:
                cross_event = True
                cross_direction = "BEARISH_CROSS"

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
