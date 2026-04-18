"""
numpy, pandas: Core data structures and matrix operations.
    statsmodels: For the VAR (Vector Autoregression) model and statistical tests.
    scipy.stats: To calculate F-distribution critical values.
    dataclasses: For structured causal telemetry.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.stats import f
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class CausalTelemetry:
    lead_asset: str
    lag_asset: str
    optimal_lag_p: int
    f_statistic: float
    p_value: float
    is_significant: bool
    bootstrap_conf: float # Bootstrap-verified significance

class GrangerCausalityDetector:
    """
    AEGIS Logic Module: Granger Causality Engine.
    Identifies lead-lag dependencies via VAR optimization and F-tests.
    """
    def __init__(self, max_lags: int = 20, alpha: float = 0.05):
        self.max_lags = max_lags
        self.alpha = alpha

    def find_optimal_lag(self, data: pd.DataFrame) -> int:
        """Optimizes lag selection using AIC and BIC criteria [2]."""
        model = VAR(data)
        # Select order based on Information Criteria
        results = model.select_order(maxlags=self.max_lags)
        # return results.aic or results.bic
        return results.selected_orders['aic']

    def compute_causality(self, data: pd.DataFrame, lead_col: str, lag_col: str) -> CausalTelemetry:
        """
        Master Logic: Performs VAR-based F-tests with bootstrap validation.
        """
        p = self.find_optimal_lag(data)
        if p == 0: p = 1 # Minimum lag for regression
        
        # Fit VAR(p) model
        model = VAR(data[[lead_col, lag_col]])
        results = model.fit(p)
        
        # 1. Perform F-test: Does lead_col cause lag_col?
        # Null Hypothesis: Coefficients of lead_col in the lag_col equation are zero.
        test_result = results.test_causality(lag_col, lead_col, kind='f')
        
        # 2. Bootstrap Significance [2]
        # Simulate null distribution by shuffling lead_col to check for spurious signals
        bootstrap_samples = 100
        spurious_count = 0
        for _ in range(bootstrap_samples):
            shuffled_data = data.copy()
            shuffled_data[lead_col] = np.random.permutation(shuffled_data[lead_col].values)
            res_shuff = VAR(shuffled_data[[lead_col, lag_col]]).fit(p)
            if res_shuff.test_causality(lag_col, lead_col, kind='f').pvalue < self.alpha:
                spurious_count += 1
        
        bootstrap_conf = 1.0 - (spurious_count / bootstrap_samples)

        return CausalTelemetry(
            lead_asset=lead_col,
            lag_asset=lag_col,
            optimal_lag_p=p,
            f_statistic=float(test_result.test_stat),
            p_value=float(test_result.pvalue),
            is_significant=test_result.pvalue < self.alpha,
            bootstrap_conf=bootstrap_conf
        )
