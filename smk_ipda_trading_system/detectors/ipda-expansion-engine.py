import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ExpansionTelemetry:
    is_expanding: bool
    direction: str         # BULLISH, BEARISH, or NONE
    lambda_1_exhaustion: float  # Volatility Ratio (Vt / ATR20)
    lambda_6_displacement: bool
    target_dol: float      # Draw on Liquidity (ERL/IRL)
    status: str

class ExpansionDetector:
    """
    Sovereign Market Kernel: Expansion Detection Engine.
    Fuses λ1 (Phase Entrapment) and λ6 (Displacement Veto) to perceive 
    institutional delivery before visual breakouts.
    """
    def __init__(self, k_multiplier: float = 1.2, delta_threshold: float = 0.7):
        self.k = k_multiplier         # Volatility ceiling for λ6
        self.delta = delta_threshold  # Exhaustion threshold for λ1
        self.lookback = 20

    def calculate_lambda_1(self, df: pd.DataFrame) -> float:
        """
        Quantifies Consolidation Volatility Decay.
        Trigger: Vt / ATR20 < 0.7 (Mathematical exhaustion).
        """
        recent = df.tail(self.lookback)
        # Price Variation Integral: sum of absolute price changes
        v_t = recent['close'].diff().abs().sum()
        atr = df['atr'].iloc[-1]
        
        return v_t / (atr + 1e-9)

    def validate_lambda_6(self, candle: dict, atr: float) -> Dict:
        """
        Direction-Aware Displacement Veto.
        Validates institutional intent via body ratio and range.
        """
        c_range = candle['high'] - candle['low']
        body = abs(candle['close'] - candle['open'])
        body_ratio = body / (c_range + 1e-9)
        
        # Constraints: Range > k*ATR AND Body Ratio > 75%
        is_displaced = (c_range > (self.k * atr)) and (body_ratio > 0.75)
        
        direction = "NONE"
        if is_displaced:
            # Bullish: Close in upper 30%; Bearish: Close in lower 30%
            if (candle['close'] - candle['low']) / (c_range + 1e-9) > 0.7:
                direction = "BULLISH"
            elif (candle['close'] - candle['low']) / (c_range + 1e-9) < 0.3:
                direction = "BEARISH"
                
        return {"active": is_displaced, "direction": direction}

    def scan(self, df: pd.DataFrame) -> ExpansionTelemetry:
        """
        Master Logic: Detects the transition from Accumulation to Distribution.
        """
        # 1. Update IPDA Structural Context (ATR & Ranges)
        atr20 = df['atr'].iloc[-1]
        l1_ratio = self.calculate_lambda_1(df)
        
        # 2. Check current candle for Displacement
        current_candle = df.iloc[-1].to_dict()
        l6_check = self.validate_lambda_6(current_candle, atr20)
        
        # 3. Predict Draw on Liquidity (DOL)
        # Target opposite Swing High/Low or FVG (IRL/ERL)
        target = df['high'].tail(60).max() if l6_check['direction'] == "BULLISH" else df['low'].tail(60).min()
        
        # 4. State Synthesis
        is_expanding = l6_check['active'] and (l1_ratio < self.delta)
        status = "EXPANSION_CONFIRMED" if is_expanding else "CONSOLIDATION_STASIS"
        
        return ExpansionTelemetry(
            is_expanding=is_expanding,
            direction=l6_check['direction'],
            lambda_1_exhaustion=round(l1_ratio, 4),
            lambda_6_displacement=l6_check['active'],
            target_dol=float(target),
            status=status
        )

