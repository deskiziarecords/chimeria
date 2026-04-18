"""
numpy: Vectorized calculation of institutional high/low nodes and the mean equilibrium.
pandas: Management of time-series OHLCV history for synchronized 20, 40, and 60-period lookbacks.
dataclasses: For structured telemetry of structural zones.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ZoneTelemetry:
    current_price: float
    equilibrium: float
    zone: str              # PREMIUM, DISCOUNT, or EQUILIBRIUM
    distance_to_eq: float  # Absolute distance in pips/points
    lookback_ranges: Dict  # H/L for 20, 40, 60 periods
    is_valid: bool         # Range structure integrity check

class PremiumDiscountDetector:
    """
    IPDA Layer 1 Component: Structural Arena Mapping.
    Identifies if price is in Premium (Short focus) or Discount (Long focus) 
    relative to the absolute 50% Equilibrium level [1, 6].
    """
    def __init__(self, lookbacks=[7-9]):
        self.lookbacks = lookbacks
        self.equilibrium = None

    def _calculate_ranges(self, df: pd.DataFrame) -> Dict:
        """Calculates institutional boundaries for synchronized lookbacks [4, 10]."""
        ranges = {}
        for lb in self.lookbacks:
            ranges[lb] = {
                'high': float(df['high'].tail(lb).max()),
                'low': float(df['low'].tail(lb).min())
            }
        return ranges

    def detect_zones(self, df: pd.DataFrame) -> ZoneTelemetry:
        """
        Master Logic: Maps price to the structural address space.
        Equilibrium = mean(H60, L60) [6, 10].
        """
        current_price = float(df['close'].iloc[-1])
        ranges = self._calculate_ranges(df)
        
        # Define the structural 'Box' using the broadest constraint (L60) [4]
        h60 = ranges[9]['high']
        l60 = ranges[9]['low']
        
        # Absolute Equilibrium: The market's fair price [1, 11]
        self.equilibrium = (h60 + l60) / 2
        
        # Determine State [1, 12]
        if current_price > self.equilibrium:
            zone = "PREMIUM" # Overextended: Search for Short Sponsorship
        elif current_price < self.equilibrium:
            zone = "DISCOUNT" # On Sale: Search for Long Sponsorship
        else:
            zone = "EQUILIBRIUM"

        # Internal Metric: Range Coherence [13]
        widths = [r['high'] - r['low'] for r in ranges.values()]
        coherence = 1 - (np.std(widths) / (np.mean(widths) + 1e-9))

        return ZoneTelemetry(
            current_price=current_price,
            equilibrium=round(self.equilibrium, 5),
            zone=zone,
            distance_to_eq=abs(current_price - self.equilibrium),
            lookback_ranges=ranges,
            is_valid=coherence > 0.4 # Invalid if ranges are non-coherent [13]
        )

