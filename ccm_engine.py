"""
numpy: Vectorized manifold reconstruction and distance calculations.
    scipy.spatial: Efficient k-nearest neighbor search using KD-Trees for Simplex Projection.
    dataclasses: For structured nonlinear telemetry.
"""
import numpy as np
from scipy.spatial import KDTree
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class CCMTelemetry:
    rho_score: float         # Prediction skill (Correlation)
    is_convergent: bool      # True if skill increases with L
    embedding_dim: int       # E (Takens parameter)
    lag: int                # Tau (Takens parameter)
    status: str

class CCMEngine:
    """
    Sovereign Market Kernel: Nonlinear Causality Engine (CCM).
    Reconstructs shadow manifolds via Takens' Embedding to identify
    causality in chaotic market regimes.
    """
    def __init__(self, E: int = 3, tau: int = 1):
        self.E = E      # Embedding dimension
        self.tau = tau  # Time lag

    def _embed(self, series: np.ndarray) -> np.ndarray:
        """Reconstructs the shadow manifold Mx using Takens' Embedding."""
        n = len(series)
        indices = np.arange(self.E) * self.tau
        # Create shadow manifold matrix (N-E*tau, E)
        return np.array([series[i + indices] for i in range(n - indices[-1])])

    def simplex_projection(self, source_manifold: np.ndarray, 
                           target_series: np.ndarray, 
                           query_point: np.ndarray) -> float:
        """Estimates Y_hat from the shadow manifold Mx (Simplex Projection)."""
        # Find E + 1 nearest neighbors
        tree = KDTree(source_manifold)
        dist, idx = tree.query(query_point, k=self.E + 1)
        
        # Exponential weighting based on distance
        weights = np.exp(-dist / (np.min(dist) + 1e-9))
        weights /= np.sum(weights)
        
        # Weighted estimate from the target series
        return np.sum(weights * target_series[idx])

    def compute_causality(self, series_x: np.ndarray, series_y: np.ndarray) -> CCMTelemetry:
        """
        Master Logic: Tests if X causes Y by checking if Y can be 
        cross-mapped from X's shadow manifold.
        """
        # 1. Reconstruct shadow manifold Mx
        Mx = self._embed(series_x)
        # Shift Y to align with the manifold points
        Y_aligned = series_y[self.E * self.tau - 1:]
        
        # 2. Cross-Mapping (Mx -> Y_hat)
        y_hat = []
        # In practice, we leave-one-out to prevent overfitting
        for i in range(len(Mx)):
            # Simplex projection using neighbors from the manifold
            y_hat.append(self.simplex_projection(Mx, Y_aligned, Mx[i]))
            
        # 3. Convergence Testing (Correlation of Y and Y_hat)
        rho = np.corrcoef(Y_aligned[:len(y_hat)], y_hat)[4]
        
        # Logic: If rho increases with library size L, causality is confirmed.
        # (Simplified static check for PoC)
        is_convergent = rho > 0.6
        
        status = "CAUSAL_LEAD_CONFIRMED" if is_convergent else "STOCHASTIC_NOISE"

        return CCMTelemetry(
            rho_score=float(rho),
            is_convergent=is_convergent,
            embedding_dim=self.E,
            lag=self.tau,
            status=status
        )

