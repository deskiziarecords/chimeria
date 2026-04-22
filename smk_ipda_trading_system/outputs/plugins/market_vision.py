"""
plugins/market_vision.py
L2-ext — MarketVisionEngine + MarketSequenceAligner
SIFT keypoints + ORB descriptors + Smith-Waterman homology detection.
The two engines are wired together: Vision finds structural nodes,
Aligner confirms if current structure is a historical homolog.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from plugins import SMKPlugin


# ─────────────────────────────────────────────────────────────────────────────
# MARKET VISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class _VisionEngine:
    def __init__(self, neighborhood: int = 16, descriptor_length: int = 16,
                 match_threshold: float = 0.85):
        self.neighborhood      = neighborhood
        self.descriptor_length = descriptor_length
        self.match_threshold   = match_threshold
        self.feature_db: list[np.ndarray] = []

    def detect_keypoints(self, prices: np.ndarray, atr: np.ndarray) -> list[dict]:
        if len(prices) < 3:
            return []
        maxima = argrelextrema(prices, np.greater)[0]
        minima = argrelextrema(prices, np.less)[0]
        candidates = np.sort(np.concatenate([maxima, minima]))
        keypoints = []
        for idx in candidates:
            if idx < self.neighborhood or idx > len(prices) - self.neighborhood:
                continue
            scale       = float(atr[idx]) if idx < len(atr) else 0.001
            delta_p     = float(prices[idx] - prices[idx - 1])
            orientation = float(np.arctan2(delta_p, 1))
            s = max(0, idx - self.descriptor_length // 2)
            e = min(len(prices), idx + self.descriptor_length // 2)
            surrounding = prices[s:e]
            diffs       = np.diff(surrounding) / (scale + 1e-9)
            descriptor  = (diffs > 0).astype(int)
            if len(descriptor) < self.descriptor_length:
                descriptor = np.pad(descriptor,
                    (0, self.descriptor_length - len(descriptor)))
            else:
                descriptor = descriptor[:self.descriptor_length]
            conviction = abs(delta_p / (scale + 1e-9))
            keypoints.append({
                "idx": int(idx), "price": float(prices[idx]),
                "scale": float(scale), "orientation": float(orientation),
                "conviction": float(conviction),
                "descriptor": descriptor,
            })
        return keypoints

    def match_and_update(self, keypoints: list[dict]) -> list[float]:
        if not keypoints or not self.feature_db:
            descs = [kp["descriptor"] for kp in keypoints]
            self.feature_db.extend(descs[-5:])
            return []
        matches = []
        for kp in keypoints:
            q = kp["descriptor"]
            for h in self.feature_db[-50:]:
                if len(q) == len(h):
                    sim = float(np.mean(q == h))
                    if sim > self.match_threshold:
                        matches.append(sim)
        descs = [kp["descriptor"] for kp in keypoints]
        self.feature_db.extend(descs[-5:])
        if len(self.feature_db) > 500:
            self.feature_db = self.feature_db[-500:]
        return matches


# ─────────────────────────────────────────────────────────────────────────────
# MARKET SEQUENCE ALIGNER  (Smith-Waterman)
# ─────────────────────────────────────────────────────────────────────────────

class _SequenceAligner:
    WEIGHTS = {'B': 9, 'I': 9, 'W': 5, 'w': 5, 'U': 3, 'D': 3, 'X': 1}

    def __init__(self, match_score: int = 2, mismatch: int = -1,
                 gap: int = -2, homology_threshold: float = 75.0):
        self.match     = match_score
        self.mismatch  = mismatch
        self.gap       = gap
        self.threshold = homology_threshold
        self._history: list[str] = []   # rolling sequence database

    @staticmethod
    def _tokenize(bar: dict) -> str:
        body  = abs(bar["close"] - bar["open"])
        rng   = max(1e-9, bar["high"] - bar["low"])
        ratio = body / rng
        wick_upper = bar["high"] - max(bar["open"], bar["close"])
        wick_lower = min(bar["open"], bar["close"]) - bar["low"]
        if ratio < 0.1:
            return "X"
        if wick_upper > body * 2 and bar["close"] < bar["open"]:
            return "W"
        if wick_lower > body * 2 and bar["close"] > bar["open"]:
            return "w"
        if bar["close"] > bar["open"]:
            return "B" if ratio > 0.6 else "U"
        return "I" if ratio > 0.6 else "D"

    def align(self, query: str, subject: str) -> dict:
        n, m = len(query), len(subject)
        if n == 0 or m == 0:
            return {"score": 0.0, "match_pct": 0.0,
                    "homology": False, "aligned": ""}
        mat = np.zeros((n + 1, m + 1))
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if query[i-1] == subject[j-1]:
                    s = self.match * self.WEIGHTS.get(query[i-1], 1)
                else:
                    s = self.mismatch
                mat[i, j] = max(0,
                    mat[i-1, j-1] + s,
                    mat[i-1, j]   + self.gap,
                    mat[i, j-1]   + self.gap)
        max_score = float(np.max(mat))
        max_poss  = min(n, m) * self.match * max(self.WEIGHTS.values())
        match_pct = float((max_score / max_poss) * 100) if max_poss > 0 else 0.0
        return {
            "score":     float(round(max_score, 2)),
            "match_pct": float(round(match_pct, 2)),
            "homology":  bool(match_pct >= self.threshold),
            "aligned":   f"{query}~{subject[:len(query)]}",
        }

    def best_historical_match(self, query: str) -> dict:
        if len(self._history) < 3 or len(query) < 3:
            return {"score": 0.0, "match_pct": 0.0, "homology": False, "aligned": ""}
        best = {"score": 0.0, "match_pct": 0.0, "homology": False, "aligned": ""}
        for hist in self._history[-30:]:
            result = self.align(query, hist)
            if result["score"] > best["score"]:
                best = result
        return best

    def update_history(self, sequence: str):
        if len(sequence) >= 4:
            self._history.append(sequence)
        if len(self._history) > 200:
            self._history = self._history[-200:]


# ─────────────────────────────────────────────────────────────────────────────
# PLUGIN
# ─────────────────────────────────────────────────────────────────────────────

class MarketVisionPlugin(SMKPlugin):
    name            = "MarketVision"
    layer           = "L2-ext"
    sensor_id       = "p03"
    requires_warmup = 32

    def __init__(self):
        super().__init__()
        self._vision  = _VisionEngine()
        self._aligner = _SequenceAligner()
        self._token_buffer: list[str] = []

    def update(self, bar: dict, df: pd.DataFrame, smk: dict) -> dict:
        prices = df["close"].values
        atr    = df["atr"].values if "atr" in df.columns else (
            (df["high"] - df["low"]).rolling(14).mean().fillna(0.001).values)

        # ── Vision: keypoints + historical matching ────────────────────────
        keypoints = self._vision.detect_keypoints(prices, atr)
        matches   = self._vision.match_and_update(keypoints)
        kp_count  = int(len(keypoints))
        match_avg = float(np.mean(matches)) if matches else 0.0
        match_count = int(len(matches))

        # ── Aligner: CLM sequence + Smith-Waterman ─────────────────────────
        token = _SequenceAligner._tokenize(bar)
        self._token_buffer.append(token)
        if len(self._token_buffer) > 32:
            self._token_buffer = self._token_buffer[-32:]
        sequence = "".join(self._token_buffer)

        alignment = self._aligner.best_historical_match(sequence[-8:])
        self._aligner.update_history(sequence[-8:])

        # ── Conviction: top 3 keypoints by conviction ─────────────────────
        top_kps = sorted(keypoints, key=lambda k: k["conviction"], reverse=True)[:3]
        top_convictions = [float(round(k["conviction"], 3)) for k in top_kps]

        active = bool(alignment["homology"] or match_count >= 2)
        score  = float(min(1.0, (match_avg + alignment["match_pct"] / 100) / 2))

        return {
            "keypoint_count":    kp_count,
            "match_count":       match_count,
            "match_avg":         float(round(match_avg, 3)),
            "top_convictions":   top_convictions,
            "sequence":          str(sequence[-12:]),
            "current_token":     str(token),
            "alignment_score":   float(alignment["score"]),
            "alignment_pct":     float(alignment["match_pct"]),
            "homology_detected": bool(alignment["homology"]),
            "status":            "HOMOLOG_CONFIRMED" if alignment["homology"] else "SCANNING",
            "active":            active,
            "score":             float(round(score, 3)),
        }
