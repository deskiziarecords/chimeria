"""
SMK Pipeline — wraps all detector modules into a single bar-by-bar processor.
This is the bridge between the FastAPI server and the actual SMK modules.
"""
import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from dataclasses import asdict
import traceback

# ── PATH RESOLUTION ──────────────────────────────────────────────────────────
# backend/ lives inside smk_ipda_trading_system/backend/
# The SMK modules (core/, lambda_sensors/, etc.) live in smk_ipda_trading_system/
# We need to add smk_ipda_trading_system/ to sys.path so imports resolve.

def _resolve_smk_root() -> str:
    """
    Walk up from this file to find the smk_ipda_trading_system root.
    Handles both layouts:
      Layout A:  smk_ipda_trading_system/backend/smk_pipeline.py  (modules are ../core etc.)
      Layout B:  quimeria/backend/smk_pipeline.py                  (modules are ../../smk_ipda_trading_system/core etc.)
    """
    here = os.path.dirname(os.path.abspath(__file__))

    # Layout A: this file IS inside smk_ipda_trading_system/backend/
    # parent dir should have core/, lambda_sensors/ etc.
    parent = os.path.dirname(here)
    if os.path.isdir(os.path.join(parent, "core")):
        return parent

    # Layout B: standalone quimeria/backend/ — SMK is a sibling folder
    # Check common sibling names
    for sibling in ["smk_ipda_trading_system", "smk", "SMK"]:
        candidate = os.path.join(parent, sibling)
        if os.path.isdir(os.path.join(candidate, "core")):
            return candidate

    # Layout C: env var override
    env_dir = os.environ.get("SMK_DIR", "")
    if env_dir and os.path.isdir(os.path.join(env_dir, "core")):
        return env_dir

    # Layout D: check if CWD contains smk_ipda_trading_system
    for cwd_child in ["smk_ipda_trading_system", "."]:
        candidate = os.path.join(os.getcwd(), cwd_child)
        if os.path.isdir(os.path.join(candidate, "core")):
            return candidate

    return ""  # not found — fallback mode


_SMK_ROOT = _resolve_smk_root()
if _SMK_ROOT and _SMK_ROOT not in sys.path:
    sys.path.insert(0, _SMK_ROOT)
    print(f"[SMK] Module root resolved: {_SMK_ROOT}")
else:
    print(f"[SMK] WARNING: SMK root not found — running numpy fallback mode")
    print(f"[SMK] Set SMK_DIR=C:\\path\\to\\smk_ipda_trading_system to fix this")
# ─────────────────────────────────────────────────────────────────────────────


class SMKPipeline:
    """
    Wraps the full Sovereign Market Kernel pipeline.
    Loads modules lazily so the server boots even if optional deps are missing.
    """

    def __init__(self):
        self.raw_bars: List[Dict] = []
        self.cursor: int = 0
        self.running: bool = False
        self.amd_state: str = "Accumulation"
        self.prev_energy: float = 0.0
        self.stasis_timer: int = 0
        self._init_modules()

    def _init_modules(self):
        """Import SMK modules — gracefully degrade if deps missing."""
        self.modules = {}
        self._import_errors = []

        def try_import(name, factory):
            try:
                self.modules[name] = factory()
            except Exception as e:
                self._import_errors.append(f"{name}: {e}")
                self.modules[name] = None

        try_import("bias",        lambda: _import("core.bias_detector",        "BiasDetector")())
        try_import("ipda",        lambda: _import("core.ipda_phase_detector",   "IPDACompiler")())
        try_import("dealing",     lambda: _import("core.dealing_range_detector","DealingRangeDetector")())
        try_import("eq_cross",    lambda: _import("core.equilibrium_cross_detector","EquilibriumCrossDetector")())
        try_import("swing",       lambda: _import("core.swing_detector",        "SwingDetector")(lookback=5))
        try_import("session",     lambda: _import("core.session_detector",      "SessionKillZoneDetector")())
        try_import("vol_decay",   lambda: _import("lambda_sensors.volatility_decay_detector","VolatilityDecayDetector")())
        try_import("displacement",lambda: _import("lambda_sensors.displacement_detector","DisplacementDetector")())
        try_import("harmonic",    lambda: _import("lambda_sensors.harmonic_trap_detector","HarmonicTrapDetector")())
        try_import("expansion",   lambda: _import("lambda_sensors.expansion_predictor","IPDAExpansionPredictor")())
        try_import("manipulation",lambda: _import("lambda_sensors.manipulation_detector","ManipulationPhaseDetector")())
        try_import("fvg",         lambda: _import("liquidity.fvg_detector_engine","FVGDetectorEngine")())
        try_import("ob",          lambda: _import("liquidity.order_block_detector","OrderBlockDetector")())
        try_import("vol_profile", lambda: _import("liquidity.volume_profile_memory_engine","VolumeProfileMemoryEngine")())
        try_import("kl",          lambda: _import("detectors.kl_divergence_detector","KLDivergenceDetector")(threshold=0.65))
        try_import("fusion",      lambda: _import("lambda_fusion_engine",       "LambdaFusionEngine")())
        try_import("mandra",      lambda: _import("risk.mandra_kernels",        "MandraGate")())

        # Topology (ripser optional)
        try_import("topology",    lambda: _import("detectors.topological_fracture_detector","TopologicalFractureDetector")())

        if self._import_errors:
            print(f"[SMK] Partial load — {len(self._import_errors)} modules failed:")
            for e in self._import_errors:
                print(f"  ⚠  {e}")
        else:
            print("[SMK] All modules loaded OK")

    def load_bars(self, bars: List[Dict]):
        self.raw_bars = bars
        self.cursor = 0
        self.amd_state = "Accumulation"
        self.prev_energy = 0.0
        self.stasis_timer = 0
        self.running = False
        # Calibrate KL reference on first 60 bars
        if self.modules.get("kl") and len(bars) >= 60:
            hist = np.array([b["close"] for b in bars[:60]])
            self.modules["kl"].calibrate_manifold(hist)

    def reset_cursor(self):
        self.cursor = 0
        self.amd_state = "Accumulation"
        self.prev_energy = 0.0
        self.stasis_timer = 0

    def get_status(self) -> Dict:
        return {
            "bars_loaded": len(self.raw_bars),
            "cursor": self.cursor,
            "amd_state": self.amd_state,
            "modules_ok": [k for k, v in self.modules.items() if v is not None],
            "modules_failed": self._import_errors,
        }

    def step(self) -> Optional[Dict]:
        """Process the next bar. Returns full detector payload or None if done."""
        if self.cursor >= len(self.raw_bars):
            return None

        idx = self.cursor
        self.cursor += 1

        W = min(60, idx + 1)
        bars_window = self.raw_bars[max(0, idx - W + 1): idx + 1]
        if len(bars_window) < 3:
            return self._minimal_payload(self.raw_bars[idx], idx)

        df = _bars_to_df(bars_window)
        cur = self.raw_bars[idx]

        result = {
            "bar": cur,
            "bar_index": idx,
            "total_bars": len(self.raw_bars),
        }

        # ── Layer 1: Structure ────────────────────────────────────────────────
        result["dealing_range"] = self._run_dealing_range(df)
        result["bias"]          = self._run_bias(df)
        result["ipda_phase"]    = self._run_ipda(df)
        result["eq_cross"]      = self._run_eq_cross(df)
        result["session"]       = self._run_session(df)
        result["swings"]        = self._run_swing(df)

        # ── Layer 2: Imbalance & Memory ───────────────────────────────────────
        result["fvg"]           = self._run_fvg(df)
        result["ob"]            = self._run_ob(df)
        result["vol_profile"]   = self._run_vol_profile(df, cur)

        # ── Lambda Sensors ────────────────────────────────────────────────────
        result["vol_decay"]     = self._run_vol_decay(df)
        result["displacement"]  = self._run_displacement(cur, df)
        result["harmonic"]      = self._run_harmonic(df)
        result["expansion"]     = self._run_expansion(df, result["dealing_range"])
        result["manipulation"]  = self._run_manipulation(df)

        # ── Detectors ─────────────────────────────────────────────────────────
        result["kl"]            = self._run_kl(df)
        result["topology"]      = self._run_topology(df)

        # ── AMD State Machine ─────────────────────────────────────────────────
        result["amd"]           = self._update_amd(result)

        # ── Ring 0: Fusion + Mandra ───────────────────────────────────────────
        result["fusion"]        = self._run_fusion(result)
        result["mandra"]        = self._run_mandra(result)
        result["veto"]          = self._compute_veto(result)

        # ── Lambda sensor summary for UI ──────────────────────────────────────
        result["sensors"]       = self._build_sensor_array(result)

        return result

    # ── INDIVIDUAL MODULE RUNNERS ─────────────────────────────────────────────

    def _run_dealing_range(self, df):
        m = self.modules.get("dealing")
        if m:
            try:
                t = m.update_ranges(df)
                if t:
                    return {"high": t.l60_high, "low": t.l60_low,
                            "eq": t.equilibrium, "zone": t.current_zone,
                            "coherence": t.coherence_score, "status": t.status}
            except: pass
        # fallback
        h = float(df["high"].max())
        l = float(df["low"].min())
        eq = (h + l) / 2
        cur = float(df["close"].iloc[-1])
        return {"high": h, "low": l, "eq": eq,
                "zone": "PREMIUM" if cur > eq else "DISCOUNT",
                "coherence": 0.8, "status": "FALLBACK"}

    def _run_bias(self, df):
        m = self.modules.get("bias")
        if m:
            try:
                t = m.detect_bias(df)
                return {"bias": t.bias, "eq": t.equilibrium, "zone": t.zone,
                        "coherence": t.coherence, "valid": t.is_valid}
            except: pass
        cur = float(df["close"].iloc[-1])
        eq = (float(df["high"].max()) + float(df["low"].min())) / 2
        return {"bias": "BULLISH" if cur > eq else "BEARISH",
                "eq": round(eq, 5), "zone": "PREMIUM" if cur > eq else "DISCOUNT",
                "coherence": 0.7, "valid": True}

    def _run_ipda(self, df):
        m = self.modules.get("ipda")
        if m:
            try:
                t = m.process_market_state(df)
                return {"phase": t.phase, "eq": float(t.equilibrium),
                        "confidence": float(t.confidence), "valid": t.is_valid}
            except: pass
        return {"phase": self.amd_state, "eq": 0.0, "confidence": 0.7, "valid": True}

    def _run_eq_cross(self, df):
        m = self.modules.get("eq_cross")
        if m:
            try:
                t = m.process_tick(df)
                return {"zone": t.zone, "cross": t.cross_event,
                        "direction": t.cross_direction, "confidence": t.confidence}
            except: pass
        return {"zone": "UNKNOWN", "cross": False, "direction": "NONE", "confidence": 0.5}

    def _run_session(self, df):
        m = self.modules.get("session")
        if m:
            try:
                ts = df.index[-1]
                t = m.detect_session(ts)
                return {"active": t.is_active, "name": t.session_name or "DEAD_ZONE",
                        "killzone": t.is_killzone, "score": t.temporal_efficiency_score,
                        "status": t.status}
            except: pass
        return {"active": False, "name": "UNKNOWN", "killzone": False, "score": 0.5, "status": "FALLBACK"}

    def _run_swing(self, df):
        m = self.modules.get("swing")
        if m and len(df) >= 10:
            try:
                swings = m.scan_pivots(df)
                return {"count": len(swings),
                        "nodes": [{"idx": s.index, "price": s.price, "type": s.type}
                                  for s in swings[-6:]]}
            except: pass
        return {"count": 0, "nodes": []}

    def _run_fvg(self, df):
        m = self.modules.get("fvg")
        if m and len(df) >= 3:
            try:
                gaps = m.scan_imbalances(df)
                recent = gaps[-3:] if gaps else []
                return {"count": len(gaps),
                        "recent": [{"type": g.gap_type, "top": g.top_boundary,
                                    "bot": g.bottom_boundary, "eq": g.equilibrium}
                                   for g in recent],
                        "active": len(gaps) > 0}
            except: pass
        return {"count": 0, "recent": [], "active": False}

    def _run_ob(self, df):
        m = self.modules.get("ob")
        if m and len(df) >= 3:
            try:
                blocks = m.scan_blocks(df)
                recent = blocks[-2:] if blocks else []
                return {"count": len(blocks),
                        "recent": [{"type": b.ob_type, "level": b.price_level,
                                    "high": b.high_boundary, "low": b.low_boundary,
                                    "score": b.displacement_score}
                                   for b in recent],
                        "active": len(blocks) > 0}
            except: pass
        return {"count": 0, "recent": [], "active": False}

    def _run_vol_profile(self, df, cur):
        m = self.modules.get("vol_profile")
        if m:
            try:
                min_p = float(df["low"].min())
                m.update_profile(cur["close"], cur["volume"], min_p)
                zones = m.detect_liquidity_zones(min_p)
                return {"zones": len(zones),
                        "hvn": [{"price": z.price_level, "density": z.density_score}
                                for z in zones if z.is_high_volume_node][:4]}
            except: pass
        return {"zones": 0, "hvn": []}

    def _run_vol_decay(self, df):
        m = self.modules.get("vol_decay")
        if m and len(df) >= 20:
            try:
                df2 = df.copy()
                if "atr20" not in df2.columns:
                    df2["atr20"] = (df2["high"] - df2["low"]).rolling(20).mean().fillna(0.001)
                t = m.detect_entrapment(df2)
                if t:
                    if t.is_entrapped: self.stasis_timer += 1
                    else: self.stasis_timer = 0
                    return {"ratio": t.volatility_ratio, "entrapped": t.is_entrapped,
                            "energy": t.latent_energy_score, "stasis": t.time_in_stasis,
                            "status": t.status}
            except: pass
        closes = df["close"].values
        vt = float(np.sum(np.abs(np.diff(closes))))
        atr = float((df["high"] - df["low"]).mean()) or 0.001
        ratio = vt / atr
        entrapped = ratio < 0.7
        if entrapped: self.stasis_timer += 1
        else: self.stasis_timer = 0
        return {"ratio": round(ratio, 4), "entrapped": entrapped,
                "energy": round(0.5 * self.stasis_timer ** 2, 2),
                "stasis": self.stasis_timer, "status": "FALLBACK"}

    def _run_displacement(self, cur, df):
        m = self.modules.get("displacement")
        if m:
            try:
                atr = float((df["high"] - df["low"]).rolling(20).mean().iloc[-1]) or 0.001
                t = m.analyze_candle(cur, atr)
                return {"is_disp": t.is_displacement, "dir": t.direction,
                        "body_ratio": t.body_ratio, "range_mult": t.range_mult,
                        "vetoed": t.is_vetoed, "status": t.status}
            except: pass
        body = abs(cur["close"] - cur["open"])
        rng = cur["high"] - cur["low"] or 0.0001
        return {"is_disp": body/rng > 0.7, "dir": 1 if cur["close"] > cur["open"] else -1,
                "body_ratio": round(body/rng, 4), "range_mult": 1.0,
                "vetoed": False, "status": "FALLBACK"}

    def _run_harmonic(self, df):
        m = self.modules.get("harmonic")
        if m and len(df) >= 64:
            try:
                actual = df["close"].values
                predicted = actual * (1 + np.sin(np.arange(len(actual)) * 0.1) * 0.001)
                t = m.detect_trap(predicted, actual)
                return {"phase_diff": t.phase_difference, "inverted": t.is_inverted,
                        "freq": t.dominant_frequency, "trap": t.trap_type, "status": t.status}
            except: pass
        closes = df["close"].values
        fft = np.fft.rfft(closes[-min(64, len(closes)):])
        phi = float(np.angle(fft[np.argmax(np.abs(fft[1:])) + 1]))
        inverted = abs(phi) > np.pi / 2
        return {"phase_diff": round(abs(phi), 3), "inverted": inverted,
                "freq": 0.0, "trap": "PHASE_INVERSION" if inverted else "NONE",
                "status": "DISSONANT: λ3 VETO" if inverted else "IN_HARMONY"}

    def _run_expansion(self, df, dr):
        m = self.modules.get("expansion")
        if m and len(df) >= 20 and "atr" in df.columns:
            try:
                magnets = {"H60": dr["high"], "L60": dr["low"], "EQ": dr["eq"]}
                t = m.predict_expansion(df, magnets)
                return {"sigma": t.sigma_t, "prob": t.expansion_prob,
                        "entrapped": t.is_entrapped, "target": t.target_dol,
                        "status": t.status}
            except: pass
        return {"sigma": 0, "prob": 0.0, "entrapped": False, "target": 0.0, "status": "FALLBACK"}

    def _run_manipulation(self, df):
        m = self.modules.get("manipulation")
        if m and len(df) >= 20:
            try:
                avg_vol = float(df["volume"].mean())
                t = m.scan_for_manipulation(df, avg_vol)
                return {"active": t.is_active, "score": t.confidence_score,
                        "level": t.sweep_level, "wick": t.wick_magnitude, "status": t.status}
            except: pass
        return {"active": False, "score": 0, "level": "NONE", "wick": 0.0, "status": "FALLBACK"}

    def _run_kl(self, df):
        m = self.modules.get("kl")
        if m:
            try:
                window = df["close"].values
                t = m.detect_drift(window)
                return {"score": t.divergence_score, "stable": t.regime_stable,
                        "h_curr": t.entropy_current, "h_ref": t.entropy_reference,
                        "status": t.status}
            except: pass
        closes = df["close"].values
        score = float(np.std(np.diff(closes)) / (np.mean(np.abs(np.diff(closes))) + 1e-9) * 0.3)
        return {"score": round(min(score, 2.0), 3), "stable": score < 0.65,
                "h_curr": 0.0, "h_ref": 0.0, "status": "FALLBACK"}

    def _run_topology(self, df):
        m = self.modules.get("topology")
        if m and len(df) >= 10:
            try:
                prices = df["close"].values
                vols = df["volume"].values
                ofi = np.diff(prices, prepend=prices[0])
                cloud = m.create_point_cloud(prices[-20:], vols[-20:], ofi[-20:])
                t = m.detect_fracture(cloud)
                return {"h1_score": t.h1_persistence_score, "fractured": t.is_fractured,
                        "islands": t.active_islands, "status": t.status}
            except: pass
        closes = df["close"].values
        score = float(np.var(np.diff(closes)) * 1e8)
        return {"h1_score": round(min(score, 10.0), 3), "fractured": score > 5.0,
                "islands": int(score / 2), "status": "COMPACT_CLOUD" if score < 5.0 else "GEOMETRY_FRACTURE"}

    def _update_amd(self, r) -> Dict:
        phase = r["ipda_phase"]["phase"]
        fvg_active = r["fvg"]["active"]
        disp_active = r["displacement"]["is_disp"]
        vol_entrapped = r["vol_decay"]["entrapped"]
        manip_active = r["manipulation"]["active"]
        kl_fractured = not r["kl"]["stable"]
        topo_fractured = r["topology"]["fractured"]

        prev = self.amd_state
        R_MASTER = kl_fractured and topo_fractured

        if R_MASTER:
            self.amd_state = "Accumulation"
        elif self.amd_state == "Accumulation":
            if vol_entrapped and r["vol_decay"]["stasis"] > 5:
                self.amd_state = "Manipulation"
        elif self.amd_state == "Manipulation":
            if manip_active or (fvg_active and disp_active):
                self.amd_state = "Distribution"
        elif self.amd_state == "Distribution":
            if vol_entrapped or r["expansion"]["sigma"] == 0:
                self.amd_state = "Retracement"
        elif self.amd_state == "Retracement":
            if not vol_entrapped and r["vol_decay"]["stasis"] == 0:
                self.amd_state = "Accumulation"

        return {"state": self.amd_state, "prev": prev,
                "changed": self.amd_state != prev, "R_MASTER": R_MASTER}

    def _run_fusion(self, r) -> Dict:
        m = self.modules.get("fusion")
        if m:
            try:
                bias = r["bias"]["bias"]
                bias_score = 1.0 if bias == "BULLISH" else -1.0 if bias == "BEARISH" else 0.0
                lambda_signals = {
                    "λ1_vol_decay": {
                        "score": 0.9 if r["vol_decay"]["entrapped"] else 0.2,
                        "confidence": min(1.0, r["vol_decay"]["energy"] / 50),
                        "veto": False
                    },
                    "λ3_harmonic": {
                        "score": -1.0 if r["harmonic"]["inverted"] else 0.4,
                        "confidence": 0.75,
                        "veto": r["harmonic"]["inverted"]
                    },
                    "λ4_manipulation": {
                        "score": 0.8 if r["manipulation"]["active"] else -0.2,
                        "confidence": r["manipulation"]["score"] / 100,
                        "veto": r["manipulation"]["active"]
                    },
                    "λ5_displacement": {
                        "score": float(r["displacement"]["dir"]),
                        "confidence": 0.85 if r["displacement"]["is_disp"] else 0.4,
                        "veto": r["displacement"]["vetoed"]
                    },
                    "λ6_bias": {
                        "score": bias_score,
                        "confidence": r["bias"]["coherence"],
                        "veto": False
                    },
                    "λ7_regime": {
                        "score": 0.5 if r["kl"]["stable"] else -0.5,
                        "confidence": 0.7,
                        "veto": not r["kl"]["stable"]
                    },
                }
                conf = r["ipda_phase"]["confidence"]
                t = m.fuse(lambda_signals=lambda_signals, ipda_phase_confidence=conf)
                return {"p_fused": t.p_fused, "confidence": t.confidence,
                        "veto_active": t.veto_active, "active_lambdas": t.active_lambdas,
                        "regime": t.regime, "status": t.status}
            except Exception as e:
                pass
        # fallback fusion
        bias = r["bias"]["bias"]
        score = 0.6 if bias == "BULLISH" else -0.6 if bias == "BEARISH" else 0.0
        if r["harmonic"]["inverted"]: score = 0.0
        return {"p_fused": round(score, 4), "confidence": r["bias"]["coherence"],
                "veto_active": r["harmonic"]["inverted"],
                "active_lambdas": [], "regime": "STABLE", "status": "FALLBACK_FUSION"}

    def _run_mandra(self, r) -> Dict:
        m = self.modules.get("mandra")
        if m:
            try:
                import numpy as np
                phi = np.array([r["fusion"]["p_fused"]])
                stab = r["fusion"]["confidence"]
                t = m.evaluate_gate(current_phi=phi, stability=stab, raw_size=0.02)
                return {"open": t.is_open, "delta_e": t.energy_delta,
                        "size": t.clamped_size, "regime_stable": t.regime_stable,
                        "status": t.status}
            except: pass
        # fallback Bit Second Law
        p = r["fusion"]["p_fused"]
        e_curr = p ** 2 * r["fusion"]["confidence"]
        de = e_curr - self.prev_energy
        self.prev_energy = e_curr
        return {"open": de >= 0, "delta_e": round(de, 4),
                "size": 0.02 if de >= 0 else 0.0, "regime_stable": True,
                "status": "GATE_OPEN" if de >= 0 else "VETO:NEGATIVE_GAIN"}

    def _compute_veto(self, r) -> Dict:
        reasons = []
        if not r["mandra"]["open"]: reasons.append("MANDRA:ΔE<0")
        if r["topology"]["fractured"]: reasons.append("TOPO:H1_FRACTURE")
        if r["fusion"]["veto_active"]: reasons.append("FUSION:LAMBDA_VETO")
        if r["harmonic"]["inverted"]: reasons.append("λ3:LIAR_STATE")
        if not r["kl"]["stable"] and r["kl"]["score"] > 1.0: reasons.append("KL:REGIME_FRACTURE")
        if r["fusion"]["confidence"] < 0.2: reasons.append("CONF:INSUFFICIENT")
        decision = "Halt" if reasons else ("Reset" if r["amd"]["R_MASTER"] else "Proceed")
        return {"decision": decision, "reasons": reasons, "trade_allowed": decision == "Proceed"}

    def _build_sensor_array(self, r) -> List[Dict]:
        vd = r["vol_decay"]
        ex = r["expansion"]
        ha = r["harmonic"]
        dr = r["dealing_range"]
        bi = r["bias"]
        di = r["displacement"]
        fv = r["fvg"]
        ob = r["ob"]
        kl = r["kl"]
        tp = r["topology"]
        ma = r["mandra"]
        se = r["session"]
        mn = r["manipulation"]
        sw = r["swings"]
        return [
            {"id": "s01", "name": "PHASE ENTRAP",  "score": vd["ratio"],           "active": vd["entrapped"]},
            {"id": "s02", "name": "EXPANSION",      "score": ex["prob"],            "active": ex["prob"] > 0.5},
            {"id": "s03", "name": "HARMONIC λ3",   "score": min(1, ha["phase_diff"] / 3.14), "active": ha["inverted"]},
            {"id": "s04", "name": "DEAL RANGE",     "score": dr["coherence"],       "active": True},
            {"id": "s05", "name": "PREM/DISC",      "score": 0.9,                   "active": dr["zone"] != "NEUTRAL"},
            {"id": "s06", "name": "DISPLACEMENT",   "score": di["body_ratio"],      "active": di["is_disp"]},
            {"id": "s07", "name": "FVG DETECT",     "score": min(1, fv["count"]/5), "active": fv["active"]},
            {"id": "s08", "name": "ORDER BLOCK",    "score": min(1, ob["count"]/5), "active": ob["active"]},
            {"id": "s09", "name": "KL DIVERGE",     "score": min(1, kl["score"]),   "active": not kl["stable"]},
            {"id": "s10", "name": "TOPO FRACT",     "score": min(1, tp["h1_score"]/5), "active": tp["fractured"]},
            {"id": "s11", "name": "MANDRA GATE",    "score": 0.9 if ma["open"] else 0.1, "active": ma["open"]},
            {"id": "s12", "name": "SESSION λ2",    "score": se["score"],           "active": se["killzone"]},
            {"id": "s13", "name": "MANIPULATION",   "score": mn["score"] / 100,    "active": mn["active"]},
            {"id": "s14", "name": "SWING NODES",    "score": min(1, sw["count"]/10), "active": sw["count"] > 0},
        ]

    def _minimal_payload(self, bar, idx):
        blank_dr = {"high": bar["high"], "low": bar["low"],
                    "eq": (bar["high"]+bar["low"])/2, "zone": "NEUTRAL",
                    "coherence": 0.5, "status": "INSUFFICIENT_DATA"}
        blank_sensors = [{"id": f"s{i:02d}", "name": "--", "score": 0.0, "active": False}
                         for i in range(1, 15)]
        return {
            "bar": bar, "bar_index": idx, "total_bars": len(self.raw_bars),
            "dealing_range": blank_dr,
            "bias": {"bias": "NEUTRAL", "eq": blank_dr["eq"], "zone": "NEUTRAL", "coherence": 0.5, "valid": False},
            "ipda_phase": {"phase": self.amd_state, "eq": blank_dr["eq"], "confidence": 0.5, "valid": False},
            "eq_cross": {"zone": "NEUTRAL", "cross": False, "direction": "NONE", "confidence": 0.5},
            "session": {"active": False, "name": "UNKNOWN", "killzone": False, "score": 0.5, "status": "INIT"},
            "swings": {"count": 0, "nodes": []},
            "fvg": {"count": 0, "recent": [], "active": False},
            "ob": {"count": 0, "recent": [], "active": False},
            "vol_profile": {"zones": 0, "hvn": []},
            "vol_decay": {"ratio": 0.0, "entrapped": False, "energy": 0.0, "stasis": 0, "status": "INIT"},
            "displacement": {"is_disp": False, "dir": 0, "body_ratio": 0.0, "range_mult": 0.0, "vetoed": False, "status": "INIT"},
            "harmonic": {"phase_diff": 0.0, "inverted": False, "freq": 0.0, "trap": "NONE", "status": "INIT"},
            "expansion": {"sigma": 0, "prob": 0.0, "entrapped": False, "target": 0.0, "status": "INIT"},
            "manipulation": {"active": False, "score": 0, "level": "NONE", "wick": 0.0, "status": "INIT"},
            "kl": {"score": 0.0, "stable": True, "h_curr": 0.0, "h_ref": 0.0, "status": "INIT"},
            "topology": {"h1_score": 0.0, "fractured": False, "islands": 0, "status": "COMPACT_CLOUD"},
            "amd": {"state": self.amd_state, "prev": self.amd_state, "changed": False, "R_MASTER": False},
            "fusion": {"p_fused": 0.0, "confidence": 0.5, "veto_active": False, "active_lambdas": [], "regime": "INIT", "status": "INIT"},
            "mandra": {"open": True, "delta_e": 0.0, "size": 0.0, "regime_stable": True, "status": "INIT"},
            "veto": {"decision": "Proceed", "reasons": [], "trade_allowed": True},
            "sensors": blank_sensors,
        }


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def _import(module_path: str, class_name: str):
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _bars_to_df(bars: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(bars)
    if "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("datetime")
    df = df.rename(columns={"open": "open", "high": "high",
                              "low": "low", "close": "close", "volume": "volume"})
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean().fillna(
                 (df["high"] - df["low"]).mean())
    df["atr20"] = (df["high"] - df["low"]).rolling(20).mean().fillna(
                   (df["high"] - df["low"]).mean())
    return df
