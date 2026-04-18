import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class OrderBlockTelemetry:
    ob_type: str        # BULLISH_OB, BEARISH_OB
    price_level: float  # Open of the origin candle
    high_boundary: float
    low_boundary: float
    is_mitigated: bool  # Has price returned to this block yet?
    displacement_score: float # Intensity of the move that followed

class OrderBlockDetector:
    """
    IPDA Layer 4 (Collector) Component: Order Block Formation Predictor.
    Identifies origin candles followed by λ6 Displacement displacement [1, 7].
    """
    def __init__(self, k_multiplier: float = 1.2, body_threshold: float = 0.75):
        self.k = k_multiplier        # Volatility ceiling (k * ATR) [1, 8]
        self.body_min = body_threshold # Minimum displacement body ratio [1, 8]

    def analyze_displacement(self, candle: dict, atr: float) -> bool:
        """Validates if a candle meets λ6 Displacement criteria [1, 9]."""
        c_range = candle['high'] - candle['low']
        body = abs(candle['close'] - candle['open'])
        body_ratio = body / (c_range + 1e-9)
        
        # λ6 Conditions: Range > k*ATR and Body Ratio > threshold
        has_range = c_range > (self.k * atr)
        has_body = body_ratio >= self.body_min
        return has_range and has_body

    def scan_blocks(self, df: pd.DataFrame) -> List[OrderBlockTelemetry]:
        """
        Scans for Bullish and Bearish Order Blocks.
        Bullish: Lowest down-close candle before expansion [1, 10].
        Bearish: Highest up-close candle before expansion [1, 10].
        """
        blocks = []
        df['tr'] = np.maximum(df['high'] - df['low'], 
                              np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                         abs(df['low'] - df['close'].shift(1))))
        df['atr20'] = df['tr'].rolling(20).mean()

        for i in range(1, len(df) - 1):
            atr = df['atr20'].iloc[i+1]
            origin_candle = df.iloc[i].to_dict()
            displacement_candle = df.iloc[i+1].to_dict()

            # 1. Bullish Order Block Detection
            # Condition: Origin is BEARISH, following is BULLISH DISPLACEMENT
            if origin_candle['close'] < origin_candle['open']:
                if displacement_candle['close'] > displacement_candle['open']:
                    if self.analyze_displacement(displacement_candle, atr):
                        blocks.append(OrderBlockTelemetry(
                            ob_type="BULLISH_OB",
                            price_level=origin_candle['open'],
                            high_boundary=origin_candle['high'],
                            low_boundary=origin_candle['low'],
                            is_mitigated=False,
                            displacement_score=displacement_candle['high'] - displacement_candle['low']
                        ))

            # 2. Bearish Order Block Detection
            # Condition: Origin is BULLISH, following is BEARISH DISPLACEMENT
            elif origin_candle['close'] > origin_candle['open']:
                if displacement_candle['close'] < displacement_candle['open']:
                    if self.analyze_displacement(displacement_candle, atr):
                        blocks.append(OrderBlockTelemetry(
                            ob_type="BEARISH_OB",
                            price_level=origin_candle['open'],
                            high_boundary=origin_candle['high'],
                            low_boundary=origin_candle['low'],
                            is_mitigated=False,
                            displacement_score=displacement_candle['high'] - displacement_candle['low']
                        ))

        return blocks
