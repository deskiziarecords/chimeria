import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class VoidTelemetry:
    price_level: float
    void_type: str  # BULLISH_VACUUM, BEARISH_VACUUM
    volume_asymmetry: float
    is_internal_liquidity: bool

class LiquidityVoidScanner:
    """
    Sovereign Market Kernel: Liquidity Void & Vacuum Block Scanner.
    Identifies asymmetric imbalance at IPDA edges [1, 7].
    """
    def __init__(self, vol_lookback: int = 20):
        self.vol_lookback = vol_lookback

    def scan_asymmetric_voids(self, df: pd.DataFrame) -> List[VoidTelemetry]:
        """
        Detects porous price action via three-candle sequence and volume asymmetry [1].
        """
        detected_voids = []
        
        # 1. Calculate historical volume baseline for asymmetry check [1]
        df['vol_avg'] = df['volume'].rolling(window=self.vol_lookback).mean()
        
        for i in range(2, len(df)):
            # 2. Sequence Detection: High of i-2 vs Low of i (Bullish Void) [1]
            is_bull_void = df['low'].iloc[i] > df['high'].iloc[i-2]
            
            # 3. Sequence Detection: Low of i-2 vs High of i (Bearish Void) [1]
            is_bear_void = df['high'].iloc[i] < df['low'].iloc[i-2]
            
            if is_bull_void or is_bear_void:
                # 4. Volume Asymmetry: current FVG volume < historical average [1]
                current_vol = df['volume'].iloc[i-1]
                vol_ratio = current_vol / (df['vol_avg'].iloc[i-1] + 1e-9)
                
                if vol_ratio < 1.0: # Institutional energy is high but participation is low [1]
                    void_type = "BULLISH_VACUUM" if is_bull_void else "BEARISH_VACUUM"
                    price_target = (df['high'].iloc[i-2] + df['low'].iloc[i]) / 2
                    
                    detected_voids.append(VoidTelemetry(
                        price_level=price_target,
                        void_type=void_type,
                        volume_asymmetry=vol_ratio,
                        is_internal_liquidity=True # Classified as IRL [2, 3]
                    ))
                    
        return detected_voids

