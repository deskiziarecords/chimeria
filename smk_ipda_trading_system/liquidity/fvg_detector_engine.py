import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class FVGTelemetry:
    gap_type: str        # BULLISH_FVG, BEARISH_FVG
    top_boundary: float
    bottom_boundary: float
    equilibrium: float   # 50% "Fair Value" of the gap
    timestamp: any

class FVGDetectorEngine:
    """
    Sovereign Market Kernel: Fair Value Gap (FVG) Detector.
    Maps the Imbalance Field (I) using a three-candle sequence validation [1, 7].
    """
    def __init__(self, sensitivity: float = 0.0):
        self.sensitivity = sensitivity

    def scan_imbalances(self, df: pd.DataFrame) -> List[FVGTelemetry]:
        """
        Identifies structural voids where price moved too fast to overlap boundaries [1].
        """
        detected_gaps = []
        highs = df['high'].values
        lows = df['low'].values
        times = df.index.values

        for i in range(2, len(df)):
            # 1. Bullish FVG: Gap between candle[i-2] high and candle[i] low [2, 7, 8]
            # Indicates upward inefficiency / liquidity vacuum
            if lows[i] > (highs[i-2] + self.sensitivity):
                detected_gaps.append(FVGTelemetry(
                    gap_type="BULLISH_FVG",
                    top_boundary=float(lows[i]),
                    bottom_boundary=float(highs[i-2]),
                    equilibrium=float((lows[i] + highs[i-2]) / 2),
                    timestamp=times[i-1]
                ))

            # 2. Bearish FVG: Gap between candle[i-2] low and candle[i] high [2, 7, 8]
            # Indicates downward inefficiency / liquidity vacuum
            elif highs[i] < (lows[i-2] - self.sensitivity):
                detected_gaps.append(FVGTelemetry(
                    gap_type="BEARISH_FVG",
                    top_boundary=float(lows[i-2]),
                    bottom_boundary=float(highs[i]),
                    equilibrium=float((lows[i-2] + highs[i]) / 2),
                    timestamp=times[i-1]
                ))

        return detected_gaps
