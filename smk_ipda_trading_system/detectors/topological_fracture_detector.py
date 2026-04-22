"""
FIXED: dgms[14] → dgms[1] (proper H1 homology dimension)
FIXED: Added infinite death handling for persistent features
Original bug: ripser returns list of 2 arrays (H0, H1), index 14 doesn't exist
"""

import numpy as np
from ripser import ripser
from dataclasses import dataclass
from typing import List

@dataclass
class TopologyTelemetry:
    h1_persistence_score: float
    is_fractured: bool
    active_islands: int
    status: str

class TopologicalFractureDetector:
    def __init__(self, persistence_threshold: float = 0.6, fracture_limit: float = 0.5):
        self.threshold = persistence_threshold
        self.fracture_limit = fracture_limit

    def create_point_cloud(self, prices: np.ndarray, volumes: np.ndarray, ofi: np.ndarray) -> np.ndarray:
        time_steps = np.arange(len(prices))
        return np.column_stack((time_steps, prices, volumes, ofi))

    def detect_fracture(self, data_cloud: np.ndarray) -> TopologyTelemetry:
        dgms = ripser(data_cloud, maxdim=1)['dgms']
        
        # FIXED: dgms[1] for H1 (1-dimensional homology), not dgms[14]
        # dgms is a list where dgms[0] = H0, dgms[1] = H1
        h1_intervals = dgms[1]  # FIXED: was dgms[14] (IndexError)
        
        if len(h1_intervals) == 0:
            return TopologyTelemetry(
                h1_persistence_score=0.0,
                is_fractured=False,
                active_islands=0,
                status="COMPACT_CLOUD"
            )
        
        # FIXED: Handle infinite lifetimes (death = inf for persistent features)
        lifetimes = []
        for birth, death in h1_intervals:
            if np.isinf(death):
                lifetimes.append(self.fracture_limit * 2)  # Treat as very persistent
            else:
                lifetimes.append(death - birth)
        
        lifetimes = np.array(lifetimes)
        distortion_score = np.sum(lifetimes)
        
        # FIXED: H0 intervals represent connected components, not H1 lifetimes
        h0_intervals = dgms[0]
        active_islands = np.sum(h0_intervals[:, 1] - h0_intervals[:, 0] > self.threshold)

        is_fractured = distortion_score > self.fracture_limit
        status = "GEOMETRY_FRACTURE" if is_fractured else "COMPACT_CLOUD"

        return TopologyTelemetry(
            h1_persistence_score=float(distortion_score),
            is_fractured=is_fractured,
            active_islands=int(active_islands),
            status=status
        )
