import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ManipulationTelemetry:
    is_active: bool
    confidence_score: float  # 0-100 heuristic score
    sweep_level: str        # H20, L40, H60, etc.
    wick_magnitude: float   # Ratio of extreme probe to body
    status: str

class ManipulationPhaseDetector:
    """
    IPDA Layer 1 (Compiler) Extension: Detects Manipulation (Judas Swings).
    Identifies institutional stop hunts via Wick Signatures at structural edges.
    """
    def __init__(self, wick_threshold: float = 3.0, threshold: int = 70):
        self.wick_threshold = wick_threshold  # wick_size > (body_size * 3) [9]
        self.anomaly_threshold = threshold
        self.state = "IDLE"

    def _calculate_ipda_ranges(self, df: pd.DataFrame) -> Dict:
        """Synchronizes 20, 40, and 60-day lookback nodes [5, 10]."""
        ranges = {}
        for lb in [11-13]:
            ranges[f'H{lb}'] = df['high'].tail(lb).max()
            ranges[f'L{lb}'] = df['low'].tail(lb).min()
        return ranges

    def detect_wick_signature(self, candle: dict) -> float:
        """Calculates the ratio of wick rejection to body size [9, 14]."""
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        wick_size = upper_wick + lower_wick
        
        # Implementation of λ6 logic: Displacement Veto/Heuristic score
        return wick_size / (body + 1e-9)

    def scan_for_manipulation(self, df: pd.DataFrame, avg_vol: float) -> ManipulationTelemetry:
        """Main execution logic for IPDA Phase 1 detection [15, 16]."""
        ranges = self._calculate_ipda_ranges(df)
        last_candle = df.iloc[-1].to_dict()
        price = last_candle['close']
        
        score = 0
        detected_level = "NONE"
        
        # 1. Edge Probe: Check if price touches or exceeds L60 boundaries [16, 17]
        for key, level in ranges.items():
            if (last_candle['high'] >= level and 'H' in key) or \
               (last_candle['low'] <= level and 'L' in key):
                score += 40
                detected_level = key
                break
        
        # 2. Wick Rejection Signature [9]
        wick_ratio = self.detect_wick_signature(last_candle)
        if wick_ratio > self.wick_threshold:
            score += 30  # Signature of Stop Hunt [9]
            
        # 3. Volume Check: Institutional Footprint (>3σ) [9]
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

