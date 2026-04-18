 """
 numpy: Vectorized ranking and matrix operations.
    pandas: Management of time-series dataframes for shifted lag analysis.
    scipy.stats: Core Spearman rank-order correlation implementation.
    dataclasses: For structured causal telemetry.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class SpearmanTelemetry:
    optimal_lag: int          # The lag τ with the highest correlation
    peak_rho: float           # Maximum Spearman Rho found
    p_value: float            # Statistical significance at peak
    conf_interval: Tuple[float, float] # 95% Bootstrap CI
    is_significant: bool      # Logic check: p < 0.05
    status: str

class SpearmanLagEngine:
    """
    Sovereign Market Kernel: Temporal Rank Correlation Engine.
    Identifies monotonic lead-lag relationships with bootstrap validation.
    """
    def __init__(self, max_lags: int = 20, bootstrap_samples: int = 100):
        self.max_lags = max_lags
        self.bootstrap_n = bootstrap_samples

    def compute_bootstrap_ci(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Calculates 95% Confidence Interval via Bootstrap resampling."""
        indices = np.arange(len(x))
        boot_rhos = []
        
        for _ in range(self.bootstrap_n):
            resample_idx = np.random.choice(indices, size=len(indices), replace=True)
            rho, _ = spearmanr(x[resample_idx], y[resample_idx])
            boot_rhos.append(rho)
            
        return float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5))

    def analyze_lead_lag(self, lead_series: pd.Series, target_series: pd.Series) -> SpearmanTelemetry:
        """
        Master Logic: Scans across temporal lags (τ) to find peak monotonic coupling.
        """
        best_rho = -1.0
        best_lag = 0
        best_p = 1.0
        
        # 1. Temporal Lag Scanning (τ)
        for lag in range(0, self.max_lags + 1):
            # Shift lead series forward in time relative to target
            shifted_lead = lead_series.shift(lag).dropna()
            aligned_target = target_series.loc[shifted_lead.index]
            
            rho, p = spearmanr(shifted_lead, aligned_target)
            
            if abs(rho) > best_rho:
                best_rho = abs(rho)
                best_lag = lag
                best_p = p
                
        # 2. Final peak series for Bootstrap
        final_lead = lead_series.shift(best_lag).dropna()
        final_target = target_series.loc[final_lead.index]
        
        # 3. Validation Layer
        ci_low, ci_high = self.compute_bootstrap_ci(final_lead.values, final_target.values)
        is_sig = best_p < 0.05 and (ci_low > 0 or ci_high < 0) # CI must not cross zero
        
        status = "CAUSAL_MONOTONIC_LEAD" if is_sig else "STOCHASTIC_DRIFT"
        
        return SpearmanTelemetry(
            optimal_lag=int(best_lag),
            peak_rho=float(best_rho),
            p_value=float(best_p),
            conf_interval=(ci_low, ci_high),
            is_significant=is_sig,
            status=status
        )

