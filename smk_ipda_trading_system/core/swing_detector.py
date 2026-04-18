
"""
    numpy: For vectorized geometric comparisons.
    pandas: For managing time-series OHLCV buffers.
    scipy.signal: For efficient local extrema detection via argrelextrema.
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SwingTelemetry:
    index: int
    price: float
    type: str  # SWING_HIGH, SWING_LOW
    strength: float
    is_liquidity_pool: bool

class SwingDetector:
    """
    Sovereign Market Kernel: Pivot High/Low Detector.
    Identifies institutional structural nodes using variable lookback windows.
    """
    def __init__(self, lookback: int = 5):
        self.lookback = lookback # Number of candles on each side

    def scan_pivots(self, df: pd.DataFrame) -> List[SwingTelemetry]:
        """
        Identifies structural pivots where high > neighbors or low < neighbors.
        Matches Source logic for Swing High/Low identification [5, 6].
        """
        swings = []
        highs = df['high'].values
        lows = df['low'].values
        
        # 1. Detect Pivot Highs (Local Maxima)
        # Condition: High[i] > High[i-n...i-1] AND High[i] > High[i+1...i+n]
        pivot_high_indices = argrelextrema(highs, np.greater, order=self.lookback)
        
        # 2. Detect Pivot Lows (Local Minima)
        # Condition: Low[i] < Low[i-n...i-1] AND Low[i] < Low[i+1...i+n]
        pivot_low_indices = argrelextrema(lows, np.less, order=self.lookback)

        # 3. Telemetry Synthesis
        for idx in pivot_high_indices:
            swings.append(SwingTelemetry(
                index=int(idx),
                price=float(highs[idx]),
                type="SWING_HIGH",
                strength=abs(highs[idx] - np.mean(highs[max(0, idx-10):idx+10])),
                is_liquidity_pool=True # Resident Buy-Side Liquidity [7]
            ))

        for idx in pivot_low_indices:
            swings.append(SwingTelemetry(
                index=int(idx),
                price=float(lows[idx]),
                type="SWING_LOW",
                strength=abs(lows[idx] - np.mean(lows[max(0, idx-10):idx+10])),
                is_liquidity_pool=True # Resident Sell-Side Liquidity [7]
            ))

        return sorted(swings, key=lambda x: x.index)
