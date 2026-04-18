"""
numpy: For vectorized range normalization and mean equilibrium calculations.
    pandas: To manage the OHLCV time-series buffers required for lookback synchronization.
    dataclasses: For structured structural telemetry.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class DealingRangeTelemetry:
    l60_high: float
    l60_low: float
    equilibrium: float    # Absolute 50% Mean [4]
    current_zone: str     # PREMIUM or DISCOUNT [3]
    coherence_score: float # Quality of range alignment
    status: str

class DealingRangeDetector:
    """
    IPDA Layer 1 (Compiler): Maps the Structural Arena.
    Identifies institutional address space via 20/40/60-period lookbacks.
    Calculates the Setup Validity Boundary (Equilibrium).
    """
    def __init__(self, lookbacks=[20, 40, 60]):
        self.lookbacks = lookbacks
        self.state = "ACCUMULATION"
        self.equilibrium = None

    def update_ranges(self, df: pd.DataFrame) -> DealingRangeTelemetry:
        """
        Synchronizes IPDA lookback nodes and calculates Equilibrium [1].
        """
        if len(df) < max(self.lookbacks):
            return None

        # 1. Map Structural Highs and Lows
        ranges = {}
        for lb in self.lookbacks:
            ranges[lb] = {
                'high': df['high'].tail(lb).max(),
                'low': df['low'].tail(lb).min()
            }

        # 2. Identify the L60 Range (Broadest Structural Constraint) [1]
        max_lb = max(self.lookbacks)
        l60_h = ranges[max_lb]['high']
        l60_l = ranges[max_lb]['low']

        # 3. Calculate Equilibrium (The Market's 'Fair Price') [3, 9]
        # Formal SMK logic uses the mean of all three lookback midpoints
        midpoints = [(ranges[lb]['high'] + ranges[lb]['low']) / 2 for lb in self.lookbacks]
        self.equilibrium = float(np.mean(midpoints))

        # 4. Determine Current Zone (Premium vs. Discount) [3]
        current_price = df['close'].iloc[-1]
        if current_price > self.equilibrium:
            zone = "PREMIUM" # Overextended: Search for Short Sponsorship
        else:
            zone = "DISCOUNT" # On Sale: Search for Long Sponsorship

        # 5. Internal Metric: Range Coherence [10]
        # (1 - variance of range widths)
        widths = [ranges[lb]['high'] - ranges[lb]['low'] for lb in self.lookbacks]
        coherence = 1 - (np.std(widths) / (np.mean(widths) + 1e-9))

        status = f"TRADING_IN_{zone}"
        if coherence < 0.4:
            status = "STRUCTURAL_FRACTURE_DETECTED"

        return DealingRangeTelemetry(
            l60_high=l60_h,
            l60_low=l60_l,
            equilibrium=round(self.equilibrium, 5),
            current_zone=zone,
            coherence_score=round(coherence, 4),
            status=status
        )

