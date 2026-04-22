"""
plugins/market_seismology.py
L3-ext — MarketSeismologyEngine (OHLCV-derived)
P/S/Surface wave classification from bar data.
P-wave: derived from volume burst density (DBSCAN approximated on OHLCV).
S-wave: body/range ratio — λ6 displacement confirmation.
Surface wave: expansion sigma + S-wave = EXPANSION_CRITICAL.
phase_pick: λ1 entrapment decay logic on price variation / ATR.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from plugins import SMKPlugin


class MarketSeismologyPlugin(SMKPlugin):
    name            = "MarketSeismology"
    layer           = "L3-ext"
    sensor_id       = "p04"
    requires_warmup = 20

    def __init__(self, p_wave_vol_sigma: float = 2.5,
                 s_wave_body_ratio: float = 0.75,
                 phase_shift_displacement: float = 1.5,
                 v_ratio_entrapment: float = 0.7):
        super().__init__()
        self.p_wave_vol_sigma         = p_wave_vol_sigma
        self.s_wave_body_ratio        = s_wave_body_ratio
        self.phase_shift_displacement = phase_shift_displacement
        self.v_ratio_entrapment       = v_ratio_entrapment

    def _detect_p_wave(self, bar: dict, df: pd.DataFrame) -> bool:
        """
        OHLCV-derived P-wave: volume burst > 2.5-sigma above mean.
        Approximates tick burst density without actual tick data.
        """
        vol     = df["volume"].values
        mean_v  = float(np.mean(vol))
        std_v   = float(np.std(vol)) + 1e-9
        z_score = (bar["volume"] - mean_v) / std_v
        return bool(z_score > self.p_wave_vol_sigma)

    def _detect_s_wave(self, bar: dict, atr: float) -> tuple[bool, float]:
        """S-wave: λ6 displacement — large body relative to range."""
        body      = abs(bar["close"] - bar["open"])
        rng       = max(1e-9, bar["high"] - bar["low"])
        body_ratio = float(body / rng)
        # Also check body > 1.2 * ATR (range expansion)
        range_mult = float(rng / (atr + 1e-9))
        is_s_wave  = bool(body_ratio > self.s_wave_body_ratio and range_mult > 1.2)
        return is_s_wave, float(round(body_ratio, 4))

    def _phase_pick(self, df: pd.DataFrame, atr: float) -> str:
        """
        λ1 phase entrapment decay: price variation / ATR20.
        Mirrors QUIMERIA's vol_decay but phrased as seismic phase.
        """
        closes = df["close"].values
        vt = float(np.sum(np.abs(np.diff(closes[-20:]))))
        v_ratio = vt / (atr + 1e-9)
        if v_ratio < self.v_ratio_entrapment:
            return "ENTRAPMENT_ZONE"
        displacement = np.abs(closes[-1] - closes)
        if np.any(displacement > self.phase_shift_displacement * atr):
            return "PHASE_SHIFT"
        return "STABLE_NOISE"

    def update(self, bar: dict, df: pd.DataFrame, smk: dict) -> dict:
        atr20 = float(df["atr20"].mean() if "atr20" in df.columns
                      else (df["high"] - df["low"]).rolling(20).mean().iloc[-1])
        atr20 = max(atr20, 1e-5)

        has_p_wave              = self._detect_p_wave(bar, df)
        has_s_wave, body_ratio  = self._detect_s_wave(bar, atr20)
        phase                   = self._phase_pick(df, atr20)

        # AMD sigma from SMK result (0=Acc, 1=Man, 2=Dis, 3=Ret)
        amd_map = {"Accumulation": 0, "Manipulation": 1,
                   "Distribution": 2, "Retracement": 3}
        sigma_t = amd_map.get(smk.get("amd", {}).get("state", "Accumulation"), 0)

        is_expanding = (sigma_t == 2 and has_s_wave)

        if is_expanding:
            # Non-linear magnitude using ATR-normalised range
            volatility_impulse = float(
                np.log1p((bar["high"] - bar["low"]) / atr20))
            magnitude  = float(np.exp(min(volatility_impulse, 3.0)))
            event_type = "SURFACE_WAVE"
            status     = "EXPANSION_CRITICAL"
        elif has_p_wave and has_s_wave:
            event_type = "S_WAVE"
            magnitude  = float(round(body_ratio, 3))
            status     = "DISPLACEMENT_CONFIRMED"
        elif has_p_wave:
            event_type = "P_WAVE"
            magnitude  = 0.35
            status     = "INSTITUTIONAL_BURST"
        elif phase == "ENTRAPMENT_ZONE":
            event_type = "ENTRAPMENT"
            magnitude  = 0.1
            status     = "PRE_QUAKE_STASIS"
        else:
            event_type = "NOISE"
            magnitude  = 0.0
            status     = "NOMINAL"

        is_epicenter = bool(is_expanding or (has_p_wave and has_s_wave))
        score        = float(min(1.0, magnitude))
        active       = bool(event_type in ("SURFACE_WAVE", "S_WAVE", "P_WAVE"))

        return {
            "event_type":    str(event_type),
            "magnitude":     float(round(magnitude, 4)),
            "is_epicenter":  is_epicenter,
            "has_p_wave":    bool(has_p_wave),
            "has_s_wave":    bool(has_s_wave),
            "body_ratio":    float(round(body_ratio, 4)),
            "phase":         str(phase),
            "sigma_t":       int(sigma_t),
            "status":        str(status),
            "active":        active,
            "score":         float(round(score, 3)),
        }
