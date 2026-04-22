"""
Port of SMK's SMART-EXE sequence engine to Python
7-symbol pattern recognition (B, I, W, w, U, D, X)
"""

import numpy as np
from typing import List, Tuple
from enum import IntEnum

class SymbolType(IntEnum):
    SYM_B = 0   # Strong Bullish
    SYM_I = 1   # Strong Bearish
    SYM_W = 2   # Upper Wick (reversal)
    SYM_w = 3   # Lower Wick (reversal)
    SYM_U = 4   # Weak Bullish
    SYM_D = 5   # Weak Bearish
    SYM_X = 6   # Neutral

SYM_CHAR = ['B', 'I', 'W', 'w', 'U', 'D', 'X']
SYM_VALUE = [900, -900, 500, -500, 330, -320, 100]
SYM_SL_PCT = [0.008, 0.008, 0.006, 0.006, 0.010, 0.010, 0.005]

# Position tables from SMART-EXE (64 positions per symbol)
POSITION_TABLES = [
    # B table (long sequence)
    [-20,-15,-10,-5,-5,-10,-15,-20,-10,0,0,5,5,0,0,-10,
     -10,5,10,15,15,10,5,-10,-5,0,15,20,20,15,0,-5,
     -5,5,15,25,25,15,5,-5,-10,0,10,20,20,10,0,-10,
     10,20,30,40,40,30,20,10,50,50,55,60,60,55,50,50],
    # I table
    [-5,-5,-5,-6,-6,-5,-5,-5,-1,-2,-3,-4,-4,-3,-2,-1,
     1,0,-1,-1,-1,-1,0,1,0,0,-1,-2,-2,-1,0,0,
     0,0,-1,-2,-2,-1,0,0,1,0,-1,-1,-1,-1,0,1,
     2,1,1,0,0,1,1,2,2,1,1,0,0,1,1,2],
    # W table
    [0,0,0,0,0,0,0,0,-1,0,0,1,1,0,0,-1,-1,0,1,2,2,1,0,-1,
     0,0,1,2,2,1,0,0,0,0,1,2,2,1,0,0,-1,0,1,1,1,1,0,-1,
     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # w table
    [0,0,0,0,0,0,0,0,1,0,-1,-1,-1,-1,0,1,0,0,-1,-2,-2,-1,0,0,
     0,0,-1,-2,-2,-1,0,0,0,0,-1,-2,-2,-1,0,0,1,0,-1,-1,-1,-1,0,1,
     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # U table
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,
     0,0,1,2,2,1,0,0,0,0,1,2,2,1,0,0,1,1,2,3,3,2,1,1,
     4,4,4,5,5,4,4,4,0,0,0,0,0,0,0,0],
    # D table
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,0,0,
     0,0,-1,-2,-2,-1,0,0,0,0,-1,-2,-2,-1,0,0,-1,-1,-2,-3,-3,-2,-1,-1,
     -4,-4,-4,-5,-5,-4,-4,-4,0,0,0,0,0,0,0,0],
    # X table
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
     0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
]

# 4D embeddings for energy/curvature calculation
EMBEDDING = [
    [ 1.0,  0.8,  0.3,  0.0],  # B
    [-1.0, -0.8, -0.3,  0.0],  # I
    [ 0.6,  0.2, -0.8,  0.5],  # W
    [-0.6, -0.2,  0.8, -0.5],  # w
    [ 0.4,  0.3,  0.1,  0.2],  # U
    [-0.4, -0.3, -0.1, -0.2],  # D
    [ 0.0,  0.0,  0.0,  0.0]   # X
]


class SMARTEXEEngine:
    """Python port of SMK's SMART-EXE sequence engine"""

    def __init__(self, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self._sequence: List[SymbolType] = [SymbolType.SYM_X] * sequence_length

    def encode_candle(self, bar: dict) -> SymbolType:
        """Encode OHLCV bar into 7-symbol alphabet"""
        o, h, l, c = float(bar['open']), float(bar['high']), float(bar['low']), float(bar['close'])
        body = abs(c - o)
        rng = h - l
        if rng < 1e-9:
            return SymbolType.SYM_X

        ratio = body / rng
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        # Wick reversals dominate
        if upper_wick > rng * 0.6:
            return SymbolType.SYM_W
        if lower_wick > rng * 0.6:
            return SymbolType.SYM_w
        if ratio < 0.10:
            return SymbolType.SYM_X

        if c > o:
            return SymbolType.SYM_B if ratio > 0.6 else SymbolType.SYM_U
        else:
            return SymbolType.SYM_I if ratio > 0.6 else SymbolType.SYM_D

    def add_bar(self, bar: dict) -> str:
        """Add bar and return current sequence string"""
        symbol = self.encode_candle(bar)
        self._sequence.pop(0)
        self._sequence.append(symbol)
        return self.get_sequence()

    def get_sequence(self) -> str:
        return ''.join(SYM_CHAR[s] for s in self._sequence)

    def evaluate_sequence(self, seq: List[SymbolType]) -> float:
        """Evaluate sequence material + position score"""
        material = 0.0
        position = 0.0
        n = len(seq)

        for i, s in enumerate(seq):
            w = (i + 1.0) / n
            material += SYM_VALUE[s] * w

            tbl_idx = int(i * 63.0 / (n - 1)) if n > 1 else 0
            position += POSITION_TABLES[s][min(63, tbl_idx)]

        return material + position

    def predict_next(self) -> Tuple[float, SymbolType]:
        """Predict next symbol and delta"""
        base = self.evaluate_sequence(self._sequence)
        best_abs = -1
        best_delta = 0
        best_sym = SymbolType.SYM_X

        for s in range(7):
            candidate = self._sequence[1:] + [SymbolType(s)]
            delta = self.evaluate_sequence(candidate) - base
            if abs(delta) > best_abs:
                best_abs = abs(delta)
                best_delta = delta
                best_sym = SymbolType(s)

        return best_delta, best_sym

    def calculate_entropy(self) -> float:
        """Shannon entropy of symbol distribution"""
        counts = [0] * 7
        for s in self._sequence:
            counts[s] += 1
        MAX_H = 2.80735  # log2(7)
        h = 0.0
        for c in counts:
            if c == 0:
                continue
            p = c / self.sequence_length
            h -= p * np.log2(p)
        return h / MAX_H

    def calculate_energy(self) -> float:
        """Geometric energy from 4D embedding differences"""
        energy = 0.0
        diff = []

        # First differences
        for i in range(len(self._sequence) - 1):
            dvec = [EMBEDDING[self._sequence[i+1]][d] - EMBEDDING[self._sequence[i]][d]
                    for d in range(4)]
            diff.append(dvec)
            energy += sum(dv * dv for dv in dvec)

        # Curvature (second differences)
        for i in range(len(diff) - 1):
            for d in range(4):
                curv = diff[i+1][d] - diff[i][d]
                energy += curv * curv

        max_e = ((len(self._sequence) - 1) + (len(self._sequence) - 2)) * 4 * 4
        return energy / max_e if max_e > 0 else 0.0

    def get_smart_metrics(self) -> dict:
        """Get all SMART-EXE metrics"""
        delta, next_sym = self.predict_next()
        return {
            "sequence": self.get_sequence(),
            "entropy": round(self.calculate_entropy(), 4),
            "energy": round(self.calculate_energy(), 4),
            "delta": round(delta, 2),
            "next_symbol": SYM_CHAR[next_sym],
            "symbol_count": {c: self._sequence.count(i)
                           for i, c in enumerate(SYM_CHAR)}
        }
