import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class LiquidityZone:
    price_level: float
    density_score: float
    is_high_volume_node: bool
    state: str # ACCUMULATING, STABLE, DEPLETED

class VolumeProfileMemoryEngine:
    """
    Layer 2 Memory Engine: Integrates Volume Profile for Liquidity Zone detection.
    Answers: Where is the hidden fuel (Time-At-Price) that will drive expansion?
    """
    def __init__(self, price_levels: int = 100, tick_size: float = 0.0001):
        self.price_levels = price_levels
        self.tick_size = tick_size
        self.time_at_price = np.zeros(price_levels) # TAP Density Array [6]
        self.hidden_liquidity = np.zeros(price_levels)
        self.state = 'ACCUMULATING'

    def _price_to_index(self, price: float, min_price: float) -> int:
        """Maps absolute price to a discrete index bucket [9]."""
        idx = int((price - min_price) / self.tick_size)
        return max(0, min(idx, self.price_levels - 1))

    def update_profile(self, current_price: float, volume: float, min_price: float):
        """Updates Time-At-Price density and decays old memory [9]."""
        price_idx = self._price_to_index(current_price, min_price)
        
        # 1. Increment Time-At-Price (TAP) density [9]
        self.time_at_price[price_idx] += 1
        
        # 2. Memory Decay (Forgetting): Prevents stale data dominance [9]
        self.time_at_price *= 0.999
        
        # 3. Fuse with Volume Profile for Hidden Liquidity estimate [10]
        self.hidden_liquidity[price_idx] += volume

    def detect_liquidity_zones(self, min_price: float) -> List[LiquidityZone]:
        """Identifies High Volume Nodes (HVNs) as structural magnets [10, 11]."""
        total_tap = np.sum(self.time_at_price) + 1e-8
        tap_density = self.time_at_price / total_tap # Normalized profile [10]
        
        zones = []
        for i, density in enumerate(tap_density):
            if density > 0.05: # Threshold for 'Significant Accumulation' [12]
                price = min_price + (i * self.tick_size)
                zones.append(LiquidityZone(
                    price_level=float(price),
                    density_score=float(density),
                    is_high_volume_node=bool(density > np.mean(tap_density) * 2),
                    state=self.state
                ))
        return zones

