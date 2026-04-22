"""
lambda_sensors/light_cone_violation.py
λ₈: Light-Cone Violation Detector - Institutional Information Leakage Capture

Catches the moment institutional information "leaks" from lead assets (DXY, SPX)
into target assets before IPDA completes delivery.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque

class ViolationType(Enum):
    INFORMATION_LEAK = "INFORMATION_LEAK"
    MOMENTUM_ANOMALY = "MOMENTUM_ANOMALY"
    SPECTRAL_INVERSION = "SPECTRAL_INVERSION"

class ViolationConfidence(Enum):
    CERTAIN = "CERTAIN"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"

@dataclass
class Lambda8Telemetry:
    score: float = 0.0
    active: bool = False
    status: str = "MONITORING"
    violation_detected: bool = False
    violation_type: str = ""
    violation_confidence: str = ""
    violation_severity: float = 0.0
    dxy_z_score: float = 0.0
    dxy_extreme: bool = False
    target_z_score: float = 0.0
    target_neutral: bool = True
    dxy_eur_correlation: float = 0.0
    kill_switch_triggered: bool = False
    kill_switch_reason: str = ""


class StochasticZScore:
    def __init__(self, period: int = 14, smooth_k: int = 3, sigma_threshold: float = 2.0):
        self.period = period
        self.smooth_k = smooth_k
        self.sigma_threshold = sigma_threshold
        self._prices: deque = deque(maxlen=period + smooth_k + 10)
        self._stochastic_history: deque = deque(maxlen=50)
    
    def update(self, price: float) -> float:
        self._prices.append(price)
        if len(self._prices) < self.period + self.smooth_k:
            return 50.0
        
        prices_list = list(self._prices)
        raw_values = []
        for i in range(min(self.smooth_k, len(prices_list) - self.period)):
            window = prices_list[-(self.period + i):-i] if i > 0 else prices_list[-self.period:]
            if len(window) >= self.period:
                recent = window
                lowest = min(recent)
                highest = max(recent)
                if highest != lowest:
                    raw_k = 100 * (recent[-1] - lowest) / (highest - lowest)
                    raw_values.append(raw_k)
        
        if not raw_values:
            return 50.0
        
        smoothed = np.mean(raw_values)
        self._stochastic_history.append(smoothed)
        return float(smoothed)
    
    def z_score(self) -> float:
        if len(self._stochastic_history) < 20:
            return 0.0
        hist = list(self._stochastic_history)
        current = hist[-1]
        mean = np.mean(hist[:-1])
        std = np.std(hist[:-1])
        if std == 0:
            return 0.0
        return float((current - mean) / std)
    
    def is_extreme(self) -> bool:
        return abs(self.z_score()) > self.sigma_threshold


class LightConeViolationDetector:
    """λ₈: Light-Cone Violation Detector"""
    
    def __init__(self, target_symbol: str = "EURUSD"):
        self.target_symbol = target_symbol
        self.dxy_stochastic = StochasticZScore(period=14, smooth_k=3)
        self.target_stochastic = StochasticZScore(period=14, smooth_k=3)
        self._price_history: Dict[str, deque] = {}
        self._warmup_bars = 30
    
    def _update_price(self, symbol: str, price: float):
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=100)
        self._price_history[symbol].append(price)
    
    def step(self,
             target_price: float,
             dxy_price: float,
             spx_price: Optional[float] = None) -> Lambda8Telemetry:
        
        telemetry = Lambda8Telemetry()
        
        self._update_price(self.target_symbol, target_price)
        self._update_price("DXY", dxy_price)
        if spx_price:
            self._update_price("SPX", spx_price)
        
        # Update stochastic indicators
        dxy_z = self.dxy_stochastic.update(dxy_price)
        target_z = self.target_stochastic.update(target_price)
        
        telemetry.dxy_z_score = dxy_z
        telemetry.dxy_extreme = self.dxy_stochastic.is_extreme()
        telemetry.target_z_score = target_z
        telemetry.target_neutral = abs(target_z) < 0.5
        
        # Calculate rolling correlation
        if len(self._price_history.get("DXY", deque())) >= 20 and len(self._price_history.get(self.target_symbol, deque())) >= 20:
            dxy_hist = list(self._price_history["DXY"])[-20:]
            target_hist = list(self._price_history[self.target_symbol])[-20:]
            corr = np.corrcoef(dxy_hist, target_hist)[0, 1]
            telemetry.dxy_eur_correlation = float(corr) if not np.isnan(corr) else 0.0
        
        # Detect momentum anomaly: DXY extreme, target neutral
        if telemetry.dxy_extreme and telemetry.target_neutral:
            telemetry.violation_detected = True
            telemetry.violation_type = ViolationType.MOMENTUM_ANOMALY.value
            telemetry.violation_confidence = ViolationConfidence.HIGH.value
            telemetry.violation_severity = min(1.0, (abs(dxy_z) - 2.0) / 2.0)
            telemetry.status = f"VIOLATION: DXY Z={dxy_z:.2f}, target neutral"
            telemetry.score = telemetry.violation_severity
            telemetry.active = True
            
            # Kill switch for severe violations
            if telemetry.violation_severity > 0.7:
                telemetry.kill_switch_triggered = True
                telemetry.kill_switch_reason = f"Light-cone violation: DXY extreme (z={dxy_z:.2f}) with neutral target"
                telemetry.status = "KILL_SWITCH_TRIGGERED"
        
        # Monitor mode
        elif telemetry.dxy_extreme:
            telemetry.score = 0.3
            telemetry.status = f"WATCHING: DXY extreme (z={dxy_z:.2f})"
        else:
            telemetry.score = 0.0
            telemetry.status = "MONITORING"
        
        return telemetry


_lambda8: Optional[LightConeViolationDetector] = None

def get_lambda8() -> LightConeViolationDetector:
    global _lambda8
    if _lambda8 is None:
        _lambda8 = LightConeViolationDetector()
    return _lambda8
