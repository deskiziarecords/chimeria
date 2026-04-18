import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class DisplacementTelemetry:
    is_displacement: bool
    direction: int         # 1 for Bullish, -1 for Bearish, 0 for None
    body_ratio: float      # Body / Range
    range_mult: float      # Range / ATR20
    is_vetoed: bool        # λ6 Conflict Status
    status: str

class DisplacementDetector:
    """
    Sovereign Market Kernel: λ6 Displacement Veto Engine.
    Identifies institutional expansion via body ratio and range multipliers.
    """
    def __init__(self, k_threshold: float = 1.2, body_ratio_min: float = 0.7):
        self.k = k_threshold        # Range volatility ceiling [4]
        self.body_min = body_ratio_min # Minimum body percentage [8]

    def analyze_candle(self, 
                       ohlc: dict, 
                       atr20: float, 
                       expected_direction: int = 0) -> DisplacementTelemetry:
        """
        Master Logic: Validates if a candle represents true institutional displacement.
        Matches Source [4, 9] specifications for λ6 Direction-Aware Veto.
        """
        # 1. Core Geometric Definitions
        candle_range = ohlc['high'] - ohlc['low']
        body = abs(ohlc['close'] - ohlc['open'])
        body_ratio = body / (candle_range + 1e-9)
        
        # 2. Displacement Constraints (D_range, D_bull, D_bear)
        # Large Range Constraint: Range > k * ATR20 [4]
        is_large_range = candle_range > (self.k * atr20)
        
        # Directional Displacement: Bullish (Upper 30%) or Bearish (Lower 30%) [9]
        is_bull_disp = (ohlc['close'] > ohlc['open']) and \
                       ((ohlc['close'] - ohlc['low']) / (candle_range + 1e-9) > 0.7)
        
        is_bear_disp = (ohlc['close'] < ohlc['open']) and \
                       ((ohlc['close'] - ohlc['low']) / (candle_range + 1e-9) < 0.3)
        
        # 3. Final State Determination
        is_disp = is_large_range and (is_bull_disp or is_bear_disp) and (body_ratio >= self.body_min)
        direction = 1 if is_bull_disp else (-1 if is_bear_disp else 0)
        
        # 4. λ6 Veto Logic: Check for conflict between micro-action and macro-intent [8, 9]
        is_vetoed = False
        if expected_direction == 1 and is_bear_disp: is_vetoed = True
        if expected_direction == -1 and is_bull_disp: is_vetoed = True

        status = "NOMINAL"
        if is_disp: status = "DISPLACEMENT_DETECTED"
        if is_vetoed: status = "HALTED: λ6 DISPLACEMENT VETO"

        return DisplacementTelemetry(
            is_displacement=is_disp,
            direction=direction,
            body_ratio=round(body_ratio, 4),
            range_mult=round(candle_range / (atr20 + 1e-9), 4),
            is_vetoed=is_vetoed,
            status=status
        )
