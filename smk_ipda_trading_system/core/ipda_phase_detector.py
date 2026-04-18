"""
numpy: Vectorized calculation of 20/40/60 ranges and equilibrium nodes.
    pandas: For managing time-series OHLCV history and lookback synchronization.
    dataclasses: For structured phase telemetry and state tracking.
""" 
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class IPDAState:
    phase: str           # ACCUMULATION, MANIPULATION, DISTRIBUTION, INVALID
    ranges: Dict         # 20/40/60 Highs and Lows
    equilibrium: float   # Absolute Mean of the structural box
    confidence: float    # Range Coherence Score
    is_valid: bool       # Structural integrity check

class IPDACompiler:
    """
    Layer 1: The Compiler.
    Defines structural context by synchronizing 20, 40, and 60-day ranges.
    Detects transition from consolidation to institutional delivery.
    """
    def __init__(self, lookbacks=[9-11]):
        self.lookbacks = lookbacks
        self.state = 'INVALID'
        self.ranges = {lb: {'high': None, 'low': None, 'mid': None} for lb in lookbacks}
        self.tau_max = 20 # Maximum time allowed in Accumulation

    def _update_ranges(self, df: pd.DataFrame):
        """Calculates institutional High/Low boundaries for all lookbacks [12, 13]."""
        for lb in self.lookbacks:
            self.ranges[lb]['high'] = df['high'].tail(lb).max()
            self.ranges[lb]['low'] = df['low'].tail(lb).min()
            self.ranges[lb]['mid'] = (self.ranges[lb]['high'] + self.ranges[lb]['low']) / 2
        
        # Absolute Equilibrium: Mean of all structural mid-points [14]
        self.equilibrium = np.mean([self.ranges[lb]['mid'] for lb in self.lookbacks])

    def detect_phase(self, current_price: float, volume: float, avg_vol: float) -> str:
        """Determines the AMD state based on structural proximity and energy [15, 16]."""
        h60, l60 = self.ranges[11]['high'], self.ranges[11]['low']
        
        # 1. Accumulation: Price hovering near Equilibrium [2, 17]
        if abs(current_price - self.equilibrium) < (0.15 * (h60 - l60)):
            # Check for exhaustion triggers (Time or Volume Collapse) [16]
            return 'ACCUMULATION'
            
        # 2. Manipulation: Sweep of range boundaries to harvest liquidity [2, 15]
        if current_price > h60 or current_price < l60:
            # Requires a 'Wick Signature' for institutional confirmation [15, 18]
            return 'MANIPULATION'
            
        # 3. Distribution: Aggressive expansion toward new Draw on Liquidity (DOL) [14, 17]
        if volume > (avg_vol * 1.5):
            return 'DISTRIBUTION'
            
        return self.state

    def process_market_state(self, df: pd.DataFrame) -> IPDAState:
        """Executes the Layer 1 state machine cycle."""
        self._update_ranges(df)
        last_candle = df.iloc[-1]
        
        # Update current phase state
        self.state = self.detect_phase(last_candle['close'], last_candle['volume'], df['volume'].mean())
        
        # Calculate Coherence Score (1 - Variance of range widths) [19]
        widths = [self.ranges[lb]['high'] - self.ranges[lb]['low'] for lb in self.lookbacks]
        coherence = 1 - (np.std(widths) / np.mean(widths))
        
        return IPDAState(
            phase=self.state,
            ranges=self.ranges,
            equilibrium=self.equilibrium,
            confidence=coherence,
            is_valid=(self.state != 'INVALID')
        )

