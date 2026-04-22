"""
plugins/kali_forensics.py
L2-ext — FileCarvingEngine + SignatureScanEngine
Foremost/Scalpel: carve accumulation zones between S/R headers+footers.
Binwalk: entropy-based trend/range detection + pattern signature scanning.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from plugins import SMKPlugin


# ─────────────────────────────────────────────────────────────────────────────
# FILE CARVING ENGINE  (Foremost/Scalpel analogy)
# ─────────────────────────────────────────────────────────────────────────────

class FileCarvingPlugin(SMKPlugin):
    """
    Treats price levels as file headers/footers.
    Carves accumulation/distribution zones between support and resistance.
    A 'carved zone' is a price region between two validated S/R levels
    where price has oscillated without trending — hidden institutional range.
    """
    name            = "FileCarvingEngine"
    layer           = "L2-ext"
    sensor_id       = "p05"
    requires_warmup = 30

    def __init__(self, zone_min_touches: int = 2, zone_width_atr: float = 0.5):
        super().__init__()
        self.zone_min_touches = zone_min_touches
        self.zone_width_atr   = zone_width_atr
        self._carved_zones: list[dict] = []   # {high, low, touches, age}

    def _find_sr_levels(self, df: pd.DataFrame, atr: float) -> list[float]:
        """Find S/R levels as price clusters with multiple touches."""
        closes   = df["close"].values
        levels   = []
        window   = atr * 0.5  # price must be within 0.5 ATR to count as same level
        for price in closes:
            matched = False
            for i, lvl in enumerate(levels):
                if abs(price - lvl["price"]) < window:
                    lvl["touches"] += 1
                    lvl["price"]    = (lvl["price"] + price) / 2  # update centroid
                    matched = True
                    break
            if not matched:
                levels.append({"price": float(price), "touches": 1})
        # Keep only levels with enough touches
        return [float(lvl["price"]) for lvl in levels
                if lvl["touches"] >= self.zone_min_touches]

    def _carve_zones(self, levels: list[float], atr: float,
                     current_price: float) -> list[dict]:
        """Carve accumulation zones between adjacent S/R levels."""
        if len(levels) < 2:
            return []
        sorted_levels = sorted(levels)
        zones = []
        for i in range(len(sorted_levels) - 1):
            low  = sorted_levels[i]
            high = sorted_levels[i + 1]
            width = high - low
            # Only carve zones wider than minimum (avoid noise)
            if width < atr * self.zone_width_atr:
                continue
            inside = bool(low <= current_price <= high)
            zones.append({
                "high":    float(round(high, 5)),
                "low":     float(round(low, 5)),
                "width":   float(round(width, 5)),
                "eq":      float(round((high + low) / 2, 5)),
                "inside":  inside,
            })
        return zones[-5:]  # top 5 zones

    def update(self, bar: dict, df: pd.DataFrame, smk: dict) -> dict:
        atr   = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.001
        atr   = max(atr, 1e-5)

        levels = self._find_sr_levels(df, atr)
        zones  = self._carve_zones(levels, atr, bar["close"])

        inside_zone = any(z["inside"] for z in zones)
        zone_count  = int(len(zones))
        level_count = int(len(levels))

        # Nearest zone boundaries
        nearest = {}
        if zones:
            current = bar["close"]
            by_dist = sorted(zones, key=lambda z: abs(z["eq"] - current))
            nearest = by_dist[0]

        active = bool(inside_zone and zone_count > 0)
        score  = float(min(1.0, zone_count / 5.0))

        return {
            "zone_count":    zone_count,
            "level_count":   level_count,
            "inside_zone":   bool(inside_zone),
            "zones":         zones[:3],   # top 3 for payload
            "nearest_high":  float(nearest.get("high", 0.0)),
            "nearest_low":   float(nearest.get("low", 0.0)),
            "nearest_eq":    float(nearest.get("eq", 0.0)),
            "status":        "INSIDE_ZONE" if inside_zone else "BETWEEN_ZONES",
            "active":        active,
            "score":         float(round(score, 3)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SIGNATURE SCAN ENGINE  (Binwalk analogy)
# ─────────────────────────────────────────────────────────────────────────────

class SignatureScanPlugin(SMKPlugin):
    """
    Scans price action for embedded pattern signatures.
    Uses entropy analysis (Binwalk-style) to classify trend vs range.
    Matches current CLM token sequence against known pattern signatures.
    """
    name            = "SignatureScanEngine"
    layer           = "L2-ext"
    sensor_id       = "p06"
    requires_warmup = 20

    # Pattern signature database  (CLM sequences → pattern name → bias)
    SIGNATURES = {
        "BBDB":    ("Bull_Flag",        1),
        "BDDB":    ("Bull_Flag",        1),
        "IXW":     ("Evening_Star",    -1),
        "BXI":     ("Doji_Reversal",   -1),
        "wXXU":    ("Accumulation",     1),
        "WXXI":    ("Distribution",    -1),
        "BBBU":    ("Strong_Uptrend",   1),
        "IIID":    ("Strong_Downtrend",-1),
        "XUBD":    ("Consolidation",    0),
        "XUXU":    ("Coil",            0),
        "WBBU":    ("Bull_Reversal",    1),
        "wIID":    ("Bear_Reversal",   -1),
        "BWBU":    ("Pullback_Long",    1),
        "IXDI":    ("Pullback_Short",  -1),
    }

    def __init__(self, entropy_trend_threshold: float = 1.5,
                 entropy_range_threshold: float = 0.6):
        super().__init__()
        self.entropy_trend = entropy_trend_threshold
        self.entropy_range = entropy_range_threshold
        self._token_buffer: list[str] = []

    @staticmethod
    def _tokenize(bar: dict) -> str:
        body  = abs(bar["close"] - bar["open"])
        rng   = max(1e-9, bar["high"] - bar["low"])
        ratio = body / rng
        wick_upper = bar["high"] - max(bar["open"], bar["close"])
        wick_lower = min(bar["open"], bar["close"]) - bar["low"]
        if ratio < 0.1: return "X"
        if wick_upper > body * 2 and bar["close"] < bar["open"]: return "W"
        if wick_lower > body * 2 and bar["close"] > bar["open"]: return "w"
        if bar["close"] > bar["open"]: return "B" if ratio > 0.6 else "U"
        return "I" if ratio > 0.6 else "D"

    def _entropy_scan(self, sequence: str) -> tuple[float, str]:
        """Binwalk entropy analysis — classify market regime."""
        if len(sequence) < 4:
            return 0.0, "INSUFFICIENT"
        chars = list(sequence)
        probs = [chars.count(c) / len(chars) for c in set(chars)]
        entropy = float(-sum(p * np.log2(p + 1e-9) for p in probs))
        if entropy > self.entropy_trend:
            regime = "RANGING"      # high entropy = random, no trend
        elif entropy < self.entropy_range:
            regime = "TRENDING"     # low entropy = concentrated, trending
        else:
            regime = "TRANSITIONAL"
        return float(round(entropy, 4)), str(regime)

    def _scan_signatures(self, sequence: str) -> dict:
        """Binwalk signature scan — match against known patterns."""
        best_match = {"pattern": "NONE", "bias": 0, "match_len": 0}
        for sig, (pattern, bias) in self.SIGNATURES.items():
            if sig in sequence:
                if len(sig) > best_match["match_len"]:
                    best_match = {"pattern": pattern, "bias": bias,
                                  "match_len": len(sig), "signature": sig}
        return best_match

    def update(self, bar: dict, df: pd.DataFrame, smk: dict) -> dict:
        token = self._tokenize(bar)
        self._token_buffer.append(token)
        if len(self._token_buffer) > 32:
            self._token_buffer = self._token_buffer[-32:]
        sequence = "".join(self._token_buffer)

        entropy, regime   = self._entropy_scan(sequence)
        sig_result        = self._scan_signatures(sequence)
        pattern_detected  = bool(sig_result["pattern"] != "NONE")
        bias              = int(sig_result.get("bias", 0))

        active = bool(pattern_detected or regime == "TRENDING")
        score  = float(min(1.0, (1.0 - entropy / 3.0) + (0.3 if pattern_detected else 0.0)))
        score  = float(round(max(0.0, min(1.0, score)), 3))

        return {
            "entropy":          float(entropy),
            "regime":           str(regime),
            "pattern":          str(sig_result["pattern"]),
            "pattern_bias":     int(bias),
            "pattern_detected": bool(pattern_detected),
            "signature":        str(sig_result.get("signature", "")),
            "sequence":         str(sequence[-12:]),
            "current_token":    str(token),
            "status":           f"{regime}:{sig_result['pattern']}",
            "active":           active,
            "score":            float(score),
        }
