"""
lambda_sensors/macro_causality_gate.py
λ₇: Macro Causality Gate - Institutional Correlation Validator

Validates trading signals against DXY, risk regimes, and SMT divergences.
Can veto trades that diverge from macro structure.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque

# ============================================================================
# TYPES & ENUMS
# ============================================================================

class CorrelationPair(Enum):
    EURUSD_DXY = "EURUSD_DXY"
    AUDUSD_GOLD = "AUDUSD_GOLD"
    SPX_DXY = "SPX_DXY"

class RegimeType(Enum):
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    TRANSITION = "TRANSITION"
    FRACTURE = "FRACTURE"

class SMTState(Enum):
    HIDDEN_BULLISH = "HIDDEN_BULLISH"
    HIDDEN_BEARISH = "HIDDEN_BEARISH"
    REGULAR_BULLISH = "REGULAR_BULLISH"
    REGULAR_BEARISH = "REGULAR_BEARISH"
    NO_DIVERGENCE = "NO_DIVERGENCE"

@dataclass
class MacroGateConfig:
    dxy_divergence_threshold: float = 0.20
    dxy_correlation_window: int = 20
    smt_lookback: int = 15
    risk_off_dxy_strength: float = 0.15
    risk_on_dxy_weakness: float = -0.15
    gate_weight: float = 0.25
    veto_enabled: bool = True

@dataclass
class Lambda7Telemetry:
    score: float = 0.0
    active: bool = False
    status: str = "IDLE"
    dxy_correlation: float = 0.0
    dxy_divergence: float = 0.0
    dxy_veto_triggered: bool = False
    smt_state: str = "NO_DIVERGENCE"
    risk_regime: str = "TRANSITION"
    liar_state: bool = False
    signal_valid: bool = True
    veto_reason: str = ""


class CorrelationEngine:
    def __init__(self, window: int = 20):
        self.window = window
        self._prices_a: deque = deque(maxlen=window)
        self._prices_b: deque = deque(maxlen=window)
    
    def update(self, price_a: float, price_b: float) -> float:
        self._prices_a.append(price_a)
        self._prices_b.append(price_b)
        if len(self._prices_a) < self.window:
            return 0.0
        arr_a = np.array(self._prices_a)
        arr_b = np.array(self._prices_b)
        corr = np.corrcoef(arr_a, arr_b)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0


class SMTDivergenceDetector:
    def __init__(self, lookback: int = 15, tolerance: float = 0.0001):
        self.lookback = lookback
        self.tolerance = tolerance
        self._prices_a: deque = deque(maxlen=lookback + 5)
        self._prices_b: deque = deque(maxlen=lookback + 5)
    
    def update(self, price_a: float, price_b: float) -> SMTState:
        self._prices_a.append(price_a)
        self._prices_b.append(price_b)
        
        if len(self._prices_a) < self.lookback:
            return SMTState.NO_DIVERGENCE
        
        recent_a = list(self._prices_a)[-self.lookback:]
        recent_b = list(self._prices_b)[-self.lookback:]
        
        # Simplified SMT detection
        last_low_a = min(recent_a[-5:])
        last_low_b = min(recent_b[-5:])
        last_high_a = max(recent_a[-5:])
        last_high_b = max(recent_b[-5:])
        
        if recent_a[-1] <= last_low_a - self.tolerance and recent_b[-1] >= last_low_b + self.tolerance:
            return SMTState.HIDDEN_BULLISH
        if recent_a[-1] >= last_high_a + self.tolerance and recent_b[-1] <= last_high_b - self.tolerance:
            return SMTState.HIDDEN_BEARISH
        
        return SMTState.NO_DIVERGENCE


class Lambda7MacroGate:
    """λ₇: Macro Causality Gate"""
    
    def __init__(self, config: Optional[MacroGateConfig] = None):
        self.config = config or MacroGateConfig()
        self.correlation_engine = CorrelationEngine(self.config.dxy_correlation_window)
        self.smt_detector = SMTDivergenceDetector(self.config.smt_lookback)
        self._price_history: Dict[str, deque] = {}
        self._warmup_bars = self.config.dxy_correlation_window
    
    def _update_price(self, asset: str, price: float):
        if asset not in self._price_history:
            self._price_history[asset] = deque(maxlen=self.config.smt_lookback + 10)
        self._price_history[asset].append(price)
    
    def step(self,
             symbol: str,
             direction: int,
             current_price: float,
             dxy_price: float,
             spx_price: Optional[float] = None,
             gold_price: Optional[float] = None) -> Lambda7Telemetry:
        
        telemetry = Lambda7Telemetry()
        
        self._update_price(symbol, current_price)
        self._update_price("DXY", dxy_price)
        if spx_price:
            self._update_price("SPX", spx_price)
        if gold_price:
            self._update_price("GOLD", gold_price)
        
        # DXY correlation
        correlation = self.correlation_engine.update(current_price, dxy_price)
        telemetry.dxy_correlation = correlation
        
        # SMT detection
        smt_state = self.smt_detector.update(current_price, dxy_price)
        telemetry.smt_state = smt_state.value
        
        # Calculate divergence
        hist_dxy = list(self._price_history.get("DXY", deque()))[-2:] if len(self._price_history.get("DXY", deque())) >= 2 else [dxy_price, dxy_price]
        dxy_change = ((hist_dxy[-1] / hist_dxy[-2]) - 1) * 100 if len(hist_dxy) == 2 else 0
        telemetry.dxy_divergence = abs(dxy_change)
        
        # Veto logic
        veto = False
        if self.config.veto_enabled and telemetry.dxy_divergence > self.config.dxy_divergence_threshold:
            if direction == 1 and dxy_change > 0:
                veto = True
                telemetry.veto_reason = "DXY divergence: long signal with DXY strength"
            elif direction == -1 and dxy_change < 0:
                veto = True
                telemetry.veto_reason = "DXY divergence: short signal with DXY weakness"
        
        telemetry.dxy_veto_triggered = veto
        telemetry.signal_valid = not veto
        
        # Risk regime
        if spx_price:
            hist_spx = list(self._price_history.get("SPX", deque()))[-2:] if len(self._price_history.get("SPX", deque())) >= 2 else [spx_price, spx_price]
            spx_change = ((hist_spx[-1] / hist_spx[-2]) - 1) * 100 if len(hist_spx) == 2 else 0
            if dxy_change > self.config.risk_off_dxy_strength and spx_change < -0.10:
                telemetry.risk_regime = "RISK_OFF"
            elif dxy_change < self.config.risk_on_dxy_weakness and spx_change > 0.10:
                telemetry.risk_regime = "RISK_ON"
            else:
                telemetry.risk_regime = "TRANSITION"
        
        # Final score
        telemetry.active = not veto
        telemetry.score = 0.8 if not veto else 0.0
        telemetry.status = "VALIDATED" if not veto else f"VETOED: {telemetry.veto_reason}"
        
        return telemetry


# Singleton for SMK pipeline
_lambda7: Optional[Lambda7MacroGate] = None

def get_lambda7() -> Lambda7MacroGate:
    global _lambda7
    if _lambda7 is None:
        _lambda7 = Lambda7MacroGate()
    return _lambda7
