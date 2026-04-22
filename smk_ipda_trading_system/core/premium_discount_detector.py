"""
FIXED: lookbacks=[7-9] → [20, 40, 60]
FIXED: ranges[9] KeyError → ranges[max_lb] (dynamic largest lookback)
Original bug: Would crash with KeyError because 9 doesn't exist in ranges dict
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ZoneTelemetry:
    current_price: float
    equilibrium: float
    zone: str
    distance_to_eq: float
    lookback_ranges: Dict
    is_valid: bool

class PremiumDiscountDetector:
    def __init__(self, lookbacks=[20, 40, 60]):  # FIXED: was [7-9] → [-2]
        self.lookbacks = lookbacks
        self.equilibrium = None

    def _calculate_ranges(self, df: pd.DataFrame) -> Dict:
        ranges = {}
        for lb in self.lookbacks:
            ranges[lb] = {
                'high': float(df['high'].tail(lb).max()),
                'low': float(df['low'].tail(lb).min())
            }
        return ranges

    def detect_zones(self, df: pd.DataFrame) -> ZoneTelemetry:
        current_price = float(df['close'].iloc[-1])
        ranges = self._calculate_ranges(df)
        
        # FIXED: Use largest lookback dynamically instead of hardcoded 9
        max_lb = max(self.lookbacks)  # FIXED: was ranges[9] which crashes
        h60 = ranges[max_lb]['high']
        l60 = ranges[max_lb]['low']
        
        self.equilibrium = (h60 + l60) / 2
        
        if current_price > self.equilibrium:
            zone = "PREMIUM"
        elif current_price < self.equilibrium:
            zone = "DISCOUNT"
        else:
            zone = "EQUILIBRIUM"

        widths = [r['high'] - r['low'] for r in ranges.values()]
        coherence = 1 - (np.std(widths) / (np.mean(widths) + 1e-9))

        return ZoneTelemetry(
            current_price=current_price,
            equilibrium=round(self.equilibrium, 5),
            zone=zone,
            distance_to_eq=abs(current_price - self.equilibrium),
            lookback_ranges=ranges,
            is_valid=coherence > 0.4
        )
