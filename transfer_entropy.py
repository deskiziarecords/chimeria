import numpy as np
from scipy.stats import entropy
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class CausalSignal:
    source_to_target: float
    target_to_source: float
    net_flow: float
    p_value: float
    is_significant: bool
    status: str

class TransferEntropyEngine:
    """
    Sovereign Market Kernel: Information-Theoretic Causality Engine.
    Implements 6-bin discretization and bootstrap shuffling for significance testing.
    """
    def __init__(self, bins: int = 6, n_shuffles: int = 100):
        self.bins = bins
        self.n_shuffles = n_shuffles

    def _discretize(self, data: np.ndarray) -> np.ndarray:
        """Converts continuous price/OFI signals into 6-bin symbolic states [2, 3]."""
        # Quantize based on percentiles to ensure uniform distribution in bins
        quantiles = np.linspace(0, 1, self.bins + 1)
        bin_edges = np.quantile(data, quantiles)
        return np.digitize(data, bin_edges[1:-1])

    def _compute_te(self, source: np.ndarray, target: np.ndarray) -> float:
        """
        Calculates Transfer Entropy: T(X->Y) = H(Y_t | Y_t-1) - H(Y_t | Y_t-1, X_t-1).
        Provides the 'Information Gain' of target given source history [3, 4].
        """
        # Shift for Markov property (Order 1)
        Y_curr = target[1:]
        Y_past = target[:-1]
        X_past = source[:-1]

        # 1. Compute H(Y_curr | Y_past)
        # Joint(Y_curr, Y_past) / Marginal(Y_past)
        def conditional_entropy_1(a, b):
            joint_hist, _ = np.histogramdd((a, b), bins=self.bins)
            joint_prob = joint_hist / np.sum(joint_hist)
            marginal_prob = np.sum(joint_prob, axis=0)
            # H(A,B) - H(B)
            return entropy(joint_prob.flatten()) - entropy(marginal_prob)

        # 2. Compute H(Y_curr | Y_past, X_past)
        def conditional_entropy_2(a, b, c):
            joint_hist, _ = np.histogramdd((a, b, c), bins=self.bins)
            joint_prob = joint_hist / np.sum(joint_hist)
            marginal_prob = np.sum(joint_prob, axis=0) # sum over Y_curr
            # H(A,B,C) - H(B,C)
            return entropy(joint_prob.flatten()) - entropy(marginal_prob.flatten())

        te = conditional_entropy_1(Y_curr, Y_past) - conditional_entropy_2(Y_curr, Y_past, X_past)
        
        # Miller-Madow Bias Correction: te_corr = te + (bins_nonzero - 1) / (2 * N)
        return max(0.0, te)

    def analyze_dependency(self, series_x: np.ndarray, series_y: np.ndarray) -> CausalSignal:
        """
        Master Logic: Bi-directional TE calculation with bootstrap significance [3].
        Identifies if X causes Y or vice versa.
        """
        # Discretization to 6-bin symbolic space
        x_sym = self._discretize(series_x)
        y_sym = self._discretize(series_y)

        # Compute Directed TE
        te_x_to_y = self._compute_te(x_sym, y_sym)
        te_y_to_x = self._compute_te(y_sym, x_sym)

        # Bootstrap Shuffling for Significance (Null Hypothesis: No interaction)
        shuffled_scores = []
        for _ in range(self.n_shuffles):
            shuffled_x = np.random.permutation(x_sym)
            shuffled_scores.append(self._compute_te(shuffled_x, y_sym))
        
        p_val = np.mean(np.array(shuffled_scores) >= te_x_to_y)
        is_sig = p_val < 0.05
        
        net = te_x_to_y - te_y_to_x
        status = "CAUSAL_LEAD_DETECTED" if is_sig and net > 0 else "NO_DIRECTED_FLOW"

        return CausalSignal(
            source_to_target=float(te_x_to_y),
            target_to_source=float(te_y_to_x),
            net_flow=float(net),
            p_value=float(p_val),
            is_significant=is_sig,
            status=status
        )
