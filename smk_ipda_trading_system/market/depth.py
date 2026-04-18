# ============================================================
# MARKET DEPTH
# ============================================================

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DepthLevel:
    price: float
    volume: float
    cumulative_volume: float
    order_count: int

@dataclass
class DepthProfile:
    instrument_id: int
    bids: List[DepthLevel]
    asks: List[DepthLevel]
    imbalance: float
    timestamp_ns: int

    @staticmethod
    def from_book(book, max_depth: int) -> 'DepthProfile':
        bid_levels, ask_levels = book.top_levels(max_depth)
        
        bids = []
        cumulative = 0.0
        for level in bid_levels:
            cumulative += level[1]
            bids.append(DepthLevel(
                price=level[0],
                volume=level[1],
                cumulative_volume=cumulative,
                order_count=0
            ))
        
        asks = []
        cumulative = 0.0
        for level in ask_levels:
            cumulative += level[1]
            asks.append(DepthLevel(
                price=level[0],
                volume=level[1],
                cumulative_volume=cumulative,
                order_count=0
            ))
        
        return DepthProfile(
            instrument_id=book.instrument_id,
            bids=bids,
            asks=asks,
            imbalance=0.0,
            timestamp_ns=book.timestamp_ns
        )
