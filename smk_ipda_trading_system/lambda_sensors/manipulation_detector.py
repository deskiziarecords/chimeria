"""
FIXED: for lb in [11-13] → [-2]
Original bug: Only iterated once with lb=-2, using 2-bar lookback for sweep detection
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ManipulationTelemetry:
    is_active: bool
    confidence_score: float
    sweep_level: str
    wick_magnitude: float
    status: str

class ManipulationPhaseDetector:
    def __init__(self, wick_threshold: float = 3.0, threshold: int = 70,
                 lookbacks=[20, 40, 60]):  # FIXED: Added proper lookbacks parameter
        self.wick_threshold = wick_threshold
        self.anomaly_threshold = threshold
        self.lookbacks = lookbacks  # FIXED: Store lookbacks as instance variable
        self.state = "IDLE"

    def _calculate_ipda_ranges(self, df: pd.DataFrame) -> Dict:
        """Synchronizes 20, 40, and 60-day lookback nodes"""
        ranges = {}
        # FIXED: was [11-13] → [-2], now uses self.lookbacks
        for lb in self.lookbacks:
            ranges[f'H{lb}'] = df['high'].tail(lb).max()
            ranges[f'L{lb}'] = df['low'].tail(lb).min()
        return ranges

    def detect_wick_signature(self, candle: dict) -> float:
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        wick_size = upper_wick + lower_wick
        return wick_size / (body + 1e-9)

    def scan_for_manipulation(self, df: pd.DataFrame, avg_vol: float) -> ManipulationTelemetry:
        ranges = self._calculate_ipda_ranges(df)
        last_candle = df.iloc[-1].to_dict()
        price = last_candle['close']
        
        score = 0
        detected_level = "NONE"
        
        # FIXED: Now properly iterates over H20/L20, H40/L40, H60/L60
        for key, level in ranges.items():
            if (last_candle['high'] >= level and 'H' in key) or \
               (last_candle['low'] <= level and 'L' in key):
                score += 40
                detected_level = key
                break
        
        wick_ratio = self.detect_wick_signature(last_candle)
        if wick_ratio > self.wick_threshold:
            score += 30
            
        if last_candle['volume'] > (avg_vol * 3):
            score += 30
            
        is_active = score >= self.anomaly_threshold
        status = f"MANIPULATION_DETECTED at {detected_level}" if is_active else "STABLE"
        
        return ManipulationTelemetry(
            is_active=is_active,
            confidence_score=float(score),
            sweep_level=detected_level,
            wick_magnitude=wick_ratio,
            status=status
        )
