"""
plugins/market_heuristic.py
L3-ext — MarketHeuristicEngine
Antivirus-style heuristics: wick/vol/entropy/momentum scoring.
Static + dynamic analysis on every bar.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from plugins import SMKPlugin


class MarketHeuristicPlugin(SMKPlugin):
    name            = "MarketHeuristic"
    layer           = "L3-ext"
    sensor_id       = "p02"
    requires_warmup = 20

    def __init__(self, threshold: float = 70.0,
                 static_weight: float = 0.7, dynamic_weight: float = 0.3,
                 entropy_threshold: float = 0.85,
                 volume_spike_mult: float = 3.0,
                 wick_body_ratio: float = 3.0,
                 body_ratio_threshold: float = 0.15):
        super().__init__()
        self.threshold            = threshold
        self.static_weight        = static_weight
        self.dynamic_weight       = dynamic_weight
        self.entropy_threshold    = entropy_threshold
        self.volume_spike_mult    = volume_spike_mult
        self.wick_body_ratio      = wick_body_ratio
        self.body_ratio_threshold = body_ratio_threshold
        self._sequence_buffer: list[str] = []   # rolling CLM token sequence

    # ── static ────────────────────────────────────────────────────────────────
    def _static_score(self, bar: dict, avg_vol: float) -> tuple[float, dict]:
        score   = 0.0
        details: dict = {}

        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
        body_size   = abs(c - o)
        wick_size   = (h - max(o, c)) + (min(o, c) - l)
        total_range = h - l + 1e-9

        wr = wick_size / (body_size + 1e-9)
        details["wick_body_ratio"] = float(round(wr, 3))
        if wick_size > body_size * self.wick_body_ratio:
            score += 30.0
            details["stop_hunt"] = True

        vr = bar["volume"] / (avg_vol + 1e-9)
        details["volume_ratio"] = float(round(vr, 3))
        if bar["volume"] > avg_vol * self.volume_spike_mult:
            score += 40.0
            details["volume_spike"] = True

        br = body_size / total_range
        details["body_ratio"] = float(round(br, 3))
        if br < self.body_ratio_threshold:
            score += 20.0
            details["indecision"] = True

        mom = (c - o) / (body_size + 1e-9)
        details["momentum"] = float(round(mom, 3))
        if abs(mom) < 0.1 and body_size > 0:
            score += 10.0
            details["momentum_divergence"] = True

        return float(score), details

    # ── dynamic ───────────────────────────────────────────────────────────────
    def _dynamic_score(self, sequence: str) -> tuple[float, float]:
        if len(sequence) < 2:
            return 0.0, 0.0
        chars = list(sequence)
        probs = [chars.count(c) / len(chars) for c in set(chars)]
        entropy = float(-sum(p * np.log2(p + 1e-9) for p in probs))
        score = 100.0 if entropy > self.entropy_threshold else 0.0
        return score, float(round(entropy, 4))

    # ── tokenize ──────────────────────────────────────────────────────────────
    @staticmethod
    def _tokenize_bar(bar: dict) -> str:
        body  = abs(bar["close"] - bar["open"])
        rng   = max(1e-9, bar["high"] - bar["low"])
        ratio = body / rng
        if ratio < 0.1:  return "X"
        if bar["close"] > bar["open"]:
            return "B" if ratio > 0.6 else "U"
        return "I" if ratio > 0.6 else "D"

    # ── lambda impact ─────────────────────────────────────────────────────────
    @staticmethod
    def _lambda_impact(static_score: float, dynamic_score: float) -> str:
        if static_score > 60:   return "λ7_CRITICAL"
        if static_score > 40:   return "λ6_VETO"
        if dynamic_score > 80:  return "λ5_DECEPTION"
        return "λ1_ENTRAPMENT"

    # ── main ──────────────────────────────────────────────────────────────────
    def update(self, bar: dict, df: pd.DataFrame, smk: dict) -> dict:
        avg_vol = float(df["volume"].mean())

        # Tokenize current bar and maintain rolling sequence
        token = self._tokenize_bar(bar)
        self._sequence_buffer.append(token)
        if len(self._sequence_buffer) > 32:
            self._sequence_buffer = self._sequence_buffer[-32:]
        sequence = "".join(self._sequence_buffer)

        static_score,  s_details = self._static_score(bar, avg_vol)
        dynamic_score, entropy   = self._dynamic_score(sequence)

        total = (static_score * self.static_weight) + (dynamic_score * self.dynamic_weight)
        is_suspicious = total >= self.threshold
        threat_type   = "POLYMORPHIC" if dynamic_score > 50 else "STATIC"
        lambda_impact = self._lambda_impact(static_score, dynamic_score)

        return {
            "score":          float(round(total, 2)),
            "static_score":   float(round(static_score, 2)),
            "dynamic_score":  float(round(dynamic_score, 2)),
            "entropy":        float(entropy),
            "is_suspicious":  bool(is_suspicious),
            "threat_type":    str(threat_type),
            "lambda_impact":  str(lambda_impact),
            "sequence":       str(sequence[-16:]),  # last 16 tokens
            "current_token":  str(token),
            "status":         str(f"{threat_type}:{lambda_impact}" if is_suspicious else "NOMINAL"),
            "active":         bool(is_suspicious),
            "score_norm":     float(round(min(1.0, total / 100.0), 3)),
            **{k: (bool(v) if isinstance(v, bool) else float(v))
               for k, v in s_details.items()},
        }
