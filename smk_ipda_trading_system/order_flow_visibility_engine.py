"""
order_flow_visibility_engine.py

Sovereign Market Kernel: Order Flow Visibility Engine
Reveals the 'Bone Structure' of the market through synthetic footprint / order flow analysis.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.signal import correlate
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class OrderFlowTelemetry:
    delta: float                    # Buyers - Sellers aggression score
    imbalances: List[Dict]          # Diagonal imbalances (> threshold)
    is_absorption: bool             # High volume + low price movement = absorption
    burst_density: float            # Institutional tick clustering density
    institutional_pulse: bool       # Matched filter confirmation of algo pulse
    status: str


class OrderFlowVisibilityEngine:
    """
    Detects hidden institutional order flow using:
    - Delta (buy vs sell aggression)
    - Diagonal imbalances
    - Absorption patterns ("invisible walls")
    - Tick burst clustering (DBSCAN)
    - Matched filter for institutional pulse signature
    """

    def __init__(self, imbalance_threshold: float = 3.0, dbscan_eps: float = 50):
        self.imbalance_threshold = imbalance_threshold
        self.dbscan_eps = dbscan_eps
        
        # Classic institutional "P-Wave" / Expansion pulse signature (PO3 style)
        self.institutional_signature = np.array([0.1, 0.2, 0.5, 1.0, 0.8, 0.4, 0.1])

    def calculate_delta(self, buy_vol: float, sell_vol: float) -> float:
        """Delta = Aggressive Buyers - Aggressive Sellers"""
        return buy_vol - sell_vol

    def detect_imbalances(self, bid_vol_profile: np.ndarray, ask_vol_profile: np.ndarray) -> List[Dict]:
        """
        Diagonal imbalance detection:
        - Ask imbalance: Aggressive buyers lifting offers (Ask[i+1] >> Bid[i])
        - Bid imbalance: Aggressive sellers hitting bids (Bid[i] >> Ask[i+1])
        """
        imbalances = []
        min_len = min(len(bid_vol_profile), len(ask_vol_profile)) - 1

        for i in range(min_len):
            # Ask-side imbalance (buyers dominating)
            if ask_vol_profile[i + 1] > (bid_vol_profile[i] * self.imbalance_threshold):
                ratio = ask_vol_profile[i + 1] / (bid_vol_profile[i] + 1e-9)
                imbalances.append({
                    "type": "ASK_IMBALANCE",
                    "level": i + 1,
                    "ratio": round(ratio, 2)
                })

            # Bid-side imbalance (sellers dominating)
            if bid_vol_profile[i] > (ask_vol_profile[i + 1] * self.imbalance_threshold):
                ratio = bid_vol_profile[i] / (ask_vol_profile[i + 1] + 1e-9)
                imbalances.append({
                    "type": "BID_IMBALANCE",
                    "level": i,
                    "ratio": round(ratio, 2)
                })

        return imbalances

    def detect_absorption(self, delta: float, price_range: float, total_volume: float) -> bool:
        """
        Absorption ("Invisible Wall"):
        High aggressive volume (large |delta|) but price barely moves.
        """
        if total_volume < 1e-9:
            return False
        return (abs(delta) > (total_volume * 0.35)) and (price_range < (total_volume * 0.0008))

    def analyze_tick_bursts(self, tick_intervals: np.ndarray, prices: np.ndarray) -> float:
        """Detect institutional bursts using DBSCAN on time + price space"""
        if len(tick_intervals) < 10:
            return 0.0

        # Feature matrix: normalized time gaps + scaled price
        features = np.column_stack((
            tick_intervals,
            (prices * 10000).astype(int)   # scale price to integer ticks
        ))

        db = DBSCAN(eps=self.dbscan_eps, min_samples=5).fit(features)
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        
        # Burst density = proportion of points in clusters (not noise)
        noise_ratio = np.sum(db.labels_ == -1) / len(tick_intervals)
        burst_density = 1.0 - noise_ratio
        
        return float(burst_density)

    def process_candle_visibility(self, 
                                  ticks: pd.DataFrame, 
                                  candle_ohlc: dict) -> OrderFlowTelemetry:
        """
        Main entry point: Analyze one candle's tick-level order flow.
        
        Expected ticks DataFrame columns:
            - delta_time (time between ticks in microseconds or seconds)
            - price
            - buy_vol
            - sell_vol
            - volume
            - bid_array (optional: bid volume profile)
            - ask_array (optional: ask volume profile)
        """
        if ticks is None or len(ticks) == 0:
            return OrderFlowTelemetry(
                delta=0.0,
                imbalances=[],
                is_absorption=False,
                burst_density=0.0,
                institutional_pulse=False,
                status="NO_TICK_DATA"
            )

        # 1. Delta
        delta = self.calculate_delta(
            ticks.get('buy_vol', pd.Series(0)).sum(),
            ticks.get('sell_vol', pd.Series(0)).sum()
        )

        # 2. Absorption
        price_range = candle_ohlc.get('high', 0) - candle_ohlc.get('low', 0)
        total_vol = ticks.get('volume', pd.Series(0)).sum()
        is_absorption = self.detect_absorption(delta, price_range, total_vol)

        # 3. Tick Burst Analysis
        burst_density = self.analyze_tick_bursts(
            ticks['delta_time'].values if 'delta_time' in ticks else np.zeros(len(ticks)),
            ticks['price'].values if 'price' in ticks else np.zeros(len(ticks))
        )

        # 4. Institutional Pulse (Matched Filter)
        volume_series = ticks['volume'].values if 'volume' in ticks else np.zeros(10)
        if len(volume_series) >= len(self.institutional_signature):
            snr = np.max(correlate(volume_series, self.institutional_signature, mode='valid'))
            institutional_pulse = snr > 0.75 * np.sum(self.institutional_signature)
        else:
            institutional_pulse = False

        # 5. Diagonal Imbalances (if bid/ask arrays provided)
        bid_array = ticks.get('bid_array', np.zeros(50))
        ask_array = ticks.get('ask_array', np.zeros(50))
        imbalances = self.detect_imbalances(bid_array, ask_array)

        status = "INSTITUTIONAL_ACTIVITY_DETECTED" if (institutional_pulse or len(imbalances) > 0 or is_absorption) else "NORMAL_FLOW"

        return OrderFlowTelemetry(
            delta=round(delta, 4),
            imbalances=imbalances,
            is_absorption=is_absorption,
            burst_density=round(burst_density, 4),
            institutional_pulse=institutional_pulse,
            status=status
        )


# ─────────────────────────────────────────────────────────────────────────────
# Quick Test / Example
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = OrderFlowVisibilityEngine(imbalance_threshold=3.0)
    
    # Dummy tick data
    dummy_ticks = pd.DataFrame({
        'delta_time': np.random.exponential(0.5, 200),
        'price': np.cumsum(np.random.randn(200) * 0.0001) + 1.0850,
        'buy_vol': np.random.randint(1, 50, 200),
        'sell_vol': np.random.randint(1, 50, 200),
        'volume': np.random.randint(10, 100, 200)
    })
    
    dummy_candle = {'high': 1.0860, 'low': 1.0840}
    
    result = engine.process_candle_visibility(dummy_ticks, dummy_candle)
    print(result)
