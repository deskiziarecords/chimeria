import sys, os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

try:
    import jax_shim
except Exception:
    pass
try:
    import statsmodels_shim
except Exception:
    pass

from chimeria_smart_exe import SMARTEXEEngine
from chimeria_ml_reversal import MLReversalPredictor

def _find_smk_root():
    here = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.get("SMK_DIR", "").strip()
    if env and os.path.isdir(os.path.join(env, "core")):
        return env
    parent = os.path.dirname(here)
    if os.path.isdir(os.path.join(parent, "core")):
        return parent
    grandparent = os.path.dirname(parent)
    for name in ["smk_ipda_trading_system", "smk", "SMK"]:
        for base in [grandparent, parent]:
            c = os.path.join(base, name)
            if os.path.isdir(os.path.join(c, "core")):
                return c
    return ""

_smk_root = _find_smk_root()
if _smk_root and _smk_root not in sys.path:
    sys.path.insert(0, _smk_root)
    print(f"[SMK] Module root: {_smk_root}")
else:
    print("[SMK] WARNING: SMK root not found - numpy fallback mode")


class SMKPipeline:
    def __init__(self):
        self.raw_bars = []
        self.cursor = 0
        self.running = False
        self.amd_state = "Accumulation"
        self.prev_energy = 0.0
        self.stasis_timer = 0
        self.modules = {}
        self._import_errors = []
        self._load_modules()
        self.smart_exe = SMARTEXEEngine()
        self.ml_reversal = MLReversalPredictor(os.path.join(_backend_dir, "models", "reversal_model.json"))

    def _load_modules(self):
        def try_load(key, factory):
            try:
                self.modules[key] = factory()
            except Exception as e:
                self._import_errors.append(f"{key}: {e}")
                self.modules[key] = None

        try_load("bias",         lambda: _imp("core.bias_detector", "BiasDetector")())
        try_load("ipda",         lambda: _imp("core.ipda_phase_detector", "IPDACompiler")())
        try_load("dealing",      lambda: _imp("core.dealing_range_detector", "DealingRangeDetector")())
        try_load("eq_cross",     lambda: _imp("core.equilibrium_cross_detector", "EquilibriumCrossDetector")())
        try_load("swing",        lambda: _imp("core.swing_detector", "SwingDetector")(lookback=5))
        try_load("session",      lambda: _imp("core.session_detector", "SessionKillZoneDetector")())
        try_load("vol_decay",    lambda: _imp("lambda_sensors.volatility_decay_detector", "VolatilityDecayDetector")())
        try_load("displacement", lambda: _imp("lambda_sensors.displacement_detector", "DisplacementDetector")())
        try_load("harmonic",     lambda: _imp("lambda_sensors.harmonic_trap_detector", "HarmonicTrapDetector")())
        try_load("expansion",    lambda: _imp("lambda_sensors.expansion_predictor", "IPDAExpansionPredictor")())
        try_load("manipulation", lambda: _imp("lambda_sensors.manipulation_detector", "ManipulationPhaseDetector")())
        try_load("fvg",          lambda: _imp("liquidity.fvg_detector_engine", "FVGDetectorEngine")())
        try_load("ob",           lambda: _imp("liquidity.order_block_detector", "OrderBlockDetector")())
        try_load("vol_profile",  lambda: _imp("liquidity.volume_profile_memory_engine", "VolumeProfileMemoryEngine")())
        try_load("kl",           lambda: _imp("detectors.kl_divergence_detector", "KLDivergenceDetector")(threshold=0.65))
        try_load("fusion",       lambda: _imp("lambda_fusion_engine", "LambdaFusionEngine")())
        try_load("mandra",       lambda: _imp("risk.mandra_kernels", "MandraGate")())
        try_load("topology",     lambda: _imp("detectors.topological_fracture_detector", "TopologicalFractureDetector")())

        # New Lambda Sensors
        try_load("lambda7",      lambda: _imp("lambda_sensors.macro_causality_gate", "Lambda7MacroGate")())
        try_load("lambda8",      lambda: _imp("lambda_sensors.light_cone_violation", "LightConeViolationDetector")())

        ok = [k for k, v in self.modules.items() if v is not None]
        bad = self._import_errors
        if bad:
            print(f"[SMK] {len(ok)} modules OK, {len(bad)} failed")
            for e in bad:
                print(f"  x  {e}")
        else:
            print(f"[SMK] All {len(ok)} modules loaded OK")

    def load_bars(self, bars):
        self.raw_bars = bars
        self.cursor = 0
        self.amd_state = "Accumulation"
        self.prev_energy = 0.0
        self.stasis_timer = 0
        kl = self.modules.get("kl")
        if kl and len(bars) >= 60:
            try:
                kl.calibrate_manifold(np.array([b["close"] for b in bars[:60]]))
            except Exception:
                pass

    def reset_cursor(self):
        self.cursor = 0
        self.amd_state = "Accumulation"
        self.prev_energy = 0.0
        self.stasis_timer = 0

    def get_status(self):
        return {
            "bars_loaded": len(self.raw_bars),
            "cursor": self.cursor,
            "amd_state": self.amd_state,
            "modules_ok": [k for k, v in self.modules.items() if v is not None],
            "modules_failed": self._import_errors,
        }

    def step(self):
        if self.cursor >= len(self.raw_bars):
            return None
        idx = self.cursor
        self.cursor += 1
        W = min(60, idx + 1)
        window = self.raw_bars[max(0, idx - W + 1): idx + 1]
        if len(window) < 3:
            return self._blank(self.raw_bars[idx], idx)
        df = _to_df(window)
        cur = self.raw_bars[idx]
        r = {'bar': cur, 'bar_index': idx, 'total_bars': len(self.raw_bars)}
        r['dealing_range'] = self._dealing_range(df)
        r['bias']          = self._bias(df)
        r['ipda_phase']    = self._ipda(df)
        r['eq_cross']      = self._eq_cross(df)
        r['session']       = self._session(df)
        r['swings']        = self._swings(df)
        r['fvg']           = self._fvg(df)
        r['ob']            = self._ob(df)
        r['vol_profile']   = self._vol_profile(df, cur)
        r['vol_decay']     = self._vol_decay(df)
        r['displacement']  = self._displacement(cur, df)
        r['harmonic']      = self._harmonic(df)
        r['expansion']     = self._expansion(df, r['dealing_range'])
        r['manipulation']  = self._manipulation(df)
        r['kl']            = self._kl(df)
        r['topology']      = self._topology(df)
        r['amd']           = self._amd(r)
        r['fusion']        = self._fusion(r)
        r['mandra']        = self._mandra(r)
        r['veto']          = self._veto(r)
        r['sensors']       = self._sensors(r)

        # ── NEW: Run λ₇ Macro Gate and λ₈ Light-Cone Detector ──────────────
        # Get current direction from bias
        direction = 1 if r['bias']['bias'] == 'BULLISH' else (-1 if r['bias']['bias'] == 'BEARISH' else 0)
        self._lambda7_macro(r, cur, direction)
        self._lambda8_light_cone(r, cur)

        # ── NEW: SMART-EXE and ML Reversal ──────────────────────────────────
        if self.cursor > 20:
            self.smart_exe.add_bar(cur)
            r['smart'] = self.smart_exe.get_smart_metrics()

        if self.ml_reversal.model and len(window) >= 60:
            reversal = self.ml_reversal.predict(df)
            r['ml_reversal'] = {
                'probability': reversal.probability,
                'is_reversal': reversal.is_reversal,
                'confidence': reversal.confidence,
                'features': reversal.features
            }

        # ── Plugin layer ──────────────────────────────────────────────────────
        try:
            mgr = _get_plugins()
            if mgr:
                # Build a minimal DataFrame for plugins from the current window
                window = self.raw_bars[max(0, idx-59):idx+1]
                import pandas as _pd
                df_plugin = _pd.DataFrame(window)
                if 'time' in df_plugin.columns:
                    df_plugin['datetime'] = _pd.to_datetime(df_plugin['time'], unit='s', utc=True)
                    df_plugin = df_plugin.set_index('datetime')
                for col in ['open','high','low','close','volume']:
                    if col in df_plugin.columns:
                        df_plugin[col] = _pd.to_numeric(df_plugin[col], errors='coerce').fillna(0)
                df_plugin['atr'] = (df_plugin['high']-df_plugin['low']).rolling(14).mean().fillna(
                    (df_plugin['high']-df_plugin['low']).mean())
                df_plugin['atr20'] = (df_plugin['high']-df_plugin['low']).rolling(20).mean().fillna(
                    (df_plugin['high']-df_plugin['low']).mean())

                plugin_results = mgr.run(cur, df_plugin, r)
                r['plugins'] = plugin_results
                # Append plugin sensors to sensor list
                r['sensors'] += mgr.to_sensor_rows(plugin_results)
        except Exception as _pe:
            r['plugins'] = {}

        # ── AEGIS Execution Bridge ────────────────────────────────────────────
        try:
            bridge = _get_bridge()
            if bridge:
                # Feed ATR on every bar regardless of veto
                bridge.update_atr(cur)
                # Only run full evaluation on PROCEED bars
                if r['veto']['decision'] == 'Proceed':
                    exe = bridge.evaluate(r, self.raw_bars[:idx+1])
                    r['execution'] = exe
                else:
                    r['execution'] = {
                        "action": "HALT",
                        "reason": r['veto']['decision'],
                        "is_armed": False,
                        "lot_size": 0.0,
                        "stop_loss_price": 0.0,
                        "take_profit_price": 0.0,
                        "kelly_size": 0.0,
                        "pattern": "",
                        "dominant": "X",
                        "direction": 0,
                        "venue_allocation": [],
                        "risk_profile": "",
                        "risk_pips": 0.0,
                        "rr_ratio": 0.0,
                        "delta_e": 0.0,
                        "rev_score": 0.0,
                    }
        except Exception as _be:
            r['execution'] = {"action": "HALT", "reason": str(_be), "is_armed": False,
                              "lot_size": 0.0, "stop_loss_price": 0.0,
                              "take_profit_price": 0.0, "kelly_size": 0.0,
                              "pattern": "", "dominant": "X", "direction": 0,
                              "venue_allocation": [], "risk_profile": "", "risk_pips": 0.0,
                              "rr_ratio": 0.0, "delta_e": 0.0, "rev_score": 0.0}

        return _sanitize(r)

    def _lambda7_macro(self, r, bar, direction):
        """Run λ₇ Macro Causality Gate"""
        try:
            gate = self.modules.get("lambda7")
            if gate:
                # Need DXY price from somewhere - you'll need to add this to your bars
                dxy_price = bar.get('dxy', 105.0)  # Fetch from data source
                spx_price = bar.get('spx', 4500.0)
                
                telemetry = gate.step(
                    symbol="EURUSD",
                    direction=direction,
                    current_price=bar['close'],
                    dxy_price=dxy_price,
                    spx_price=spx_price
                )
                
                r['lambda_7'] = {
                    'score': telemetry.score,
                    'active': telemetry.active,
                    'status': telemetry.status,
                    'dxy_correlation': telemetry.dxy_correlation,
                    'dxy_veto': telemetry.dxy_veto_triggered,
                    'signal_valid': telemetry.signal_valid,
                    'risk_regime': telemetry.risk_regime
                }
                
                # Add to sensors
                r['sensors'].append({
                    'id': 'λ₇',
                    'name': 'Macro Gate',
                    'score': telemetry.score,
                    'active': telemetry.active,
                    'layer': 'L5-MACRO',
                    'status': telemetry.status
                })
                
                # Apply veto
                if telemetry.dxy_veto_triggered:
                    r['veto']['decision'] = 'Halt'
                    r['veto']['reasons'].append(f"λ₇: {telemetry.veto_reason}")
        except Exception as e:
            print(f"[SMK] λ₇ error: {e}")

    def _lambda8_light_cone(self, r, bar):
        """Run λ₈ Light-Cone Violation Detector"""
        try:
            detector = self.modules.get("lambda8")
            if detector:
                dxy_price = bar.get('dxy', 105.0)
                spx_price = bar.get('spx', 4500.0)
                
                telemetry = detector.step(
                    target_price=bar['close'],
                    dxy_price=dxy_price,
                    spx_price=spx_price
                )
                
                r['lambda_8'] = {
                    'score': telemetry.score,
                    'active': telemetry.active,
                    'status': telemetry.status,
                    'violation_detected': telemetry.violation_detected,
                    'violation_type': telemetry.violation_type,
                    'dxy_z_score': telemetry.dxy_z_score,
                    'target_z_score': telemetry.target_z_score,
                    'kill_switch': telemetry.kill_switch_triggered
                }
                
                r['sensors'].append({
                    'id': 'λ₈',
                    'name': 'Light-Cone',
                    'score': telemetry.score,
                    'active': telemetry.active,
                    'layer': 'L0-ALPHA',
                    'status': telemetry.status
                })
                
                # Kill switch overrides everything
                if telemetry.kill_switch_triggered:
                    r['veto']['decision'] = 'Halt'
                    r['veto']['reasons'].append(f"λ₈: {telemetry.kill_switch_reason}")
                    if 'execution' in r:
                        r['execution']['action'] = 'HALT'
        except Exception as e:
            print(f"[SMK] λ₈ error: {e}")

    def _dealing_range(self, df):
        m = self.modules.get("dealing")
        if m:
            try:
                t = m.update_ranges(df)
                if t:
                    return {"high": t.l60_high, "low": t.l60_low, "eq": t.equilibrium,
                            "zone": t.current_zone, "coherence": t.coherence_score, "status": t.status}
            except Exception:
                pass
        h = float(df["high"].max()); l = float(df["low"].min()); eq = (h + l) / 2
        cur = float(df["close"].iloc[-1])
        return {"high": h, "low": l, "eq": eq,
                "zone": "PREMIUM" if cur > eq else "DISCOUNT", "coherence": 0.8, "status": "FALLBACK"}

    def _bias(self, df):
        m = self.modules.get("bias")
        if m:
            try:
                t = m.detect_bias(df)
                return {"bias": t.bias, "eq": t.equilibrium, "zone": t.zone,
                        "coherence": t.coherence, "valid": t.is_valid}
            except Exception:
                pass
        cur = float(df["close"].iloc[-1]); eq = (float(df["high"].max()) + float(df["low"].min())) / 2
        return {"bias": "BULLISH" if cur > eq else "BEARISH", "eq": round(eq, 5),
                "zone": "PREMIUM" if cur > eq else "DISCOUNT", "coherence": 0.7, "valid": True}

    def _ipda(self, df):
        m = self.modules.get("ipda")
        if m:
            try:
                t = m.process_market_state(df)
                return {"phase": t.phase, "eq": float(t.equilibrium),
                        "confidence": float(t.confidence), "valid": t.is_valid}
            except Exception:
                pass
        return {"phase": self.amd_state, "eq": 0.0, "confidence": 0.7, "valid": True}

    def _eq_cross(self, df):
        m = self.modules.get("eq_cross")
        if m:
            try:
                t = m.process_tick(df)
                return {"zone": t.zone, "cross": t.cross_event,
                        "direction": t.cross_direction, "confidence": t.confidence}
            except Exception:
                pass
        return {"zone": "UNKNOWN", "cross": False, "direction": "NONE", "confidence": 0.5}

    def _session(self, df):
        m = self.modules.get("session")
        if m:
            try:
                t = m.detect_session(df.index[-1])
                return {"active": t.is_active, "name": t.session_name or "DEAD_ZONE",
                        "killzone": t.is_killzone, "score": t.temporal_efficiency_score,
                        "status": t.status}
            except Exception:
                pass
        return {"active": False, "name": "UNKNOWN", "killzone": False, "score": 0.5, "status": "FALLBACK"}

    def _swings(self, df):
        m = self.modules.get("swing")
        if m and len(df) >= 10:
            try:
                sw = m.scan_pivots(df)
                return {"count": len(sw),
                        "nodes": [{"idx": s.index, "price": s.price, "type": s.type} for s in sw[-6:]]}
            except Exception:
                pass
        return {"count": 0, "nodes": []}

    def _fvg(self, df):
        m = self.modules.get("fvg")
        if m and len(df) >= 3:
            try:
                gaps = m.scan_imbalances(df)
                return {"count": len(gaps), "active": len(gaps) > 0,
                        "recent": [{"type": g.gap_type, "top": g.top_boundary,
                                    "bot": g.bottom_boundary, "eq": g.equilibrium} for g in gaps[-3:]]}
            except Exception:
                pass
        return {"count": 0, "active": False, "recent": []}

    def _ob(self, df):
        m = self.modules.get("ob")
        if m and len(df) >= 3:
            try:
                blocks = m.scan_blocks(df)
                return {"count": len(blocks), "active": len(blocks) > 0,
                        "recent": [{"type": b.ob_type, "level": b.price_level,
                                    "high": b.high_boundary, "low": b.low_boundary,
                                    "score": b.displacement_score} for b in blocks[-2:]]}
            except Exception:
                pass
        return {"count": 0, "active": False, "recent": []}

    def _vol_profile(self, df, cur):
        m = self.modules.get("vol_profile")
        if m:
            try:
                min_p = float(df["low"].min())
                m.update_profile(cur["close"], cur["volume"], min_p)
                zones = m.detect_liquidity_zones(min_p)
                return {"zones": len(zones),
                        "hvn": [{"price": z.price_level, "density": z.density_score}
                                for z in zones if z.is_high_volume_node][:4]}
            except Exception:
                pass
        return {"zones": 0, "hvn": []}

    def _vol_decay(self, df):
        m = self.modules.get("vol_decay")
        if m and len(df) >= 20:
            try:
                df2 = df.copy()
                if "atr20" not in df2.columns:
                    df2["atr20"] = (df2["high"] - df2["low"]).rolling(20).mean().fillna(0.001)
                t = m.detect_entrapment(df2)
                if t:
                    self.stasis_timer = (self.stasis_timer + 1) if t.is_entrapped else 0
                    return {"ratio": t.volatility_ratio, "entrapped": t.is_entrapped,
                            "energy": t.latent_energy_score, "stasis": t.time_in_stasis,
                            "status": t.status}
            except Exception:
                pass
        closes = df["close"].values
        vt = float(np.sum(np.abs(np.diff(closes))))
        atr = float((df["high"] - df["low"]).mean()) or 0.001
        ratio = vt / atr
        entrapped = ratio < 0.7
        self.stasis_timer = (self.stasis_timer + 1) if entrapped else 0
        return {"ratio": round(ratio, 4), "entrapped": entrapped,
                "energy": round(0.5 * self.stasis_timer ** 2, 2),
                "stasis": self.stasis_timer, "status": "FALLBACK"}

    def _displacement(self, cur, df):
        m = self.modules.get("displacement")
        if m:
            try:
                atr = float((df["high"] - df["low"]).rolling(20).mean().iloc[-1]) or 0.001
                t = m.analyze_candle(cur, atr)
                return {"is_disp": t.is_displacement, "dir": t.direction,
                        "body_ratio": t.body_ratio, "range_mult": t.range_mult,
                        "vetoed": t.is_vetoed, "status": t.status}
            except Exception:
                pass
        body = abs(cur["close"] - cur["open"])
        rng = (cur["high"] - cur["low"]) or 0.0001
        return {"is_disp": body / rng > 0.7, "dir": 1 if cur["close"] > cur["open"] else -1,
                "body_ratio": round(body / rng, 4), "range_mult": 1.0,
                "vetoed": False, "status": "FALLBACK"}

    def _harmonic(self, df):
        m = self.modules.get("harmonic")
        if m and len(df) >= 64:
            try:
                actual = df["close"].values
                pred = actual * (1 + np.sin(np.arange(len(actual)) * 0.1) * 0.001)
                t = m.detect_trap(pred, actual)
                return {"phase_diff": t.phase_difference, "inverted": t.is_inverted,
                        "freq": t.dominant_frequency, "trap": t.trap_type, "status": t.status}
            except Exception:
                pass
        closes = df["close"].values[-min(64, len(df)):]
        fft = np.fft.rfft(closes)
        phi = float(np.angle(fft[np.argmax(np.abs(fft[1:])) + 1]))
        inv = abs(phi) > np.pi / 2
        return {"phase_diff": round(abs(phi), 3), "inverted": inv, "freq": 0.0,
                "trap": "PHASE_INVERSION" if inv else "NONE",
                "status": "DISSONANT: L3 VETO" if inv else "IN_HARMONY"}

    def _expansion(self, df, dr):
        m = self.modules.get("expansion")
        if m and len(df) >= 20 and "atr" in df.columns:
            try:
                magnets = {"H60": dr["high"], "L60": dr["low"], "EQ": dr["eq"]}
                t = m.predict_expansion(df, magnets)
                return {"sigma": t.sigma_t, "prob": t.expansion_prob,
                        "entrapped": t.is_entrapped, "target": t.target_dol, "status": t.status}
            except Exception:
                pass
        return {"sigma": 0, "prob": 0.0, "entrapped": False, "target": 0.0, "status": "FALLBACK"}

    def _manipulation(self, df):
        m = self.modules.get("manipulation")
        if m and len(df) >= 20:
            try:
                t = m.scan_for_manipulation(df, float(df["volume"].mean()))
                return {"active": t.is_active, "score": t.confidence_score,
                        "level": t.sweep_level, "wick": t.wick_magnitude, "status": t.status}
            except Exception:
                pass
        return {"active": False, "score": 0, "level": "NONE", "wick": 0.0, "status": "FALLBACK"}

    def _kl(self, df):
        m = self.modules.get("kl")
        if m:
            try:
                t = m.detect_drift(df["close"].values)
                return {"score": t.divergence_score, "stable": t.regime_stable,
                        "h_curr": t.entropy_current, "h_ref": t.entropy_reference, "status": t.status}
            except Exception:
                pass
        closes = df["close"].values
        score = float(np.std(np.diff(closes)) / (np.mean(np.abs(np.diff(closes))) + 1e-9) * 0.3)
        return {"score": round(min(score, 2.0), 3), "stable": score < 0.65,
                "h_curr": 0.0, "h_ref": 0.0, "status": "FALLBACK"}

    def _topology(self, df):
        m = self.modules.get("topology")
        if m and len(df) >= 10:
            try:
                prices = df["close"].values[-20:]
                vols = df["volume"].values[-20:]
                ofi = np.diff(prices, prepend=prices[0])
                cloud = m.create_point_cloud(prices, vols, ofi)
                t = m.detect_fracture(cloud)
                return {"h1_score": t.h1_persistence_score, "fractured": t.is_fractured,
                        "islands": t.active_islands, "status": t.status}
            except Exception:
                pass
        closes = df["close"].values
        score = float(np.var(np.diff(closes)) * 1e8)
        return {"h1_score": round(min(score, 10.0), 3), "fractured": score > 5.0,
                "islands": int(score / 2),
                "status": "COMPACT_CLOUD" if score < 5.0 else "GEOMETRY_FRACTURE"}

    def _amd(self, r):
        prev = self.amd_state
        R_MASTER = not r["kl"]["stable"] and r["topology"]["fractured"]
        if R_MASTER:
            self.amd_state = "Accumulation"
        elif self.amd_state == "Accumulation":
            if r["vol_decay"]["entrapped"] and r["vol_decay"]["stasis"] > 5:
                self.amd_state = "Manipulation"
        elif self.amd_state == "Manipulation":
            if r["manipulation"]["active"] or (r["fvg"]["active"] and r["displacement"]["is_disp"]):
                self.amd_state = "Distribution"
        elif self.amd_state == "Distribution":
            if r["vol_decay"]["entrapped"] or r["expansion"]["sigma"] == 0:
                self.amd_state = "Retracement"
        elif self.amd_state == "Retracement":
            if not r["vol_decay"]["entrapped"] and r["vol_decay"]["stasis"] == 0:
                self.amd_state = "Accumulation"
        return {"state": self.amd_state, "prev": prev,
                "changed": self.amd_state != prev, "R_MASTER": R_MASTER}

    def _fusion(self, r):
        m = self.modules.get("fusion")
        if m:
            try:
                bias_score = (1.0 if r["bias"]["bias"] == "BULLISH"
                              else -1.0 if r["bias"]["bias"] == "BEARISH" else 0.0)
                signals = {
                    "lambda1_vol_decay":    {"score": 0.9 if r["vol_decay"]["entrapped"] else 0.2,
                                             "confidence": min(1.0, r["vol_decay"]["energy"] / 50), "veto": False},
                    "lambda3_harmonic":     {"score": -1.0 if r["harmonic"]["inverted"] else 0.4,
                                             "confidence": 0.75, "veto": r["harmonic"]["inverted"]},
                    "lambda4_manipulation": {"score": 0.8 if r["manipulation"]["active"] else -0.2,
                                             "confidence": r["manipulation"]["score"] / 100,
                                             "veto": r["manipulation"]["active"]},
                    "lambda5_displacement": {"score": float(r["displacement"]["dir"]),
                                             "confidence": 0.85 if r["displacement"]["is_disp"] else 0.4,
                                             "veto": r["displacement"]["vetoed"]},
                    "lambda6_bias":         {"score": bias_score,
                                             "confidence": r["bias"]["coherence"], "veto": False},
                    "lambda7_regime":       {"score": 0.5 if r["kl"]["stable"] else -0.5,
                                             "confidence": 0.7, "veto": not r["kl"]["stable"]},
                }
                t = m.fuse(lambda_signals=signals,
                           ipda_phase_confidence=r["ipda_phase"]["confidence"])
                return {"p_fused": t.p_fused, "confidence": t.confidence,
                        "veto_active": t.veto_active, "active_lambdas": t.active_lambdas,
                        "regime": t.regime, "status": t.status}
            except Exception:
                pass
        bias = r["bias"]["bias"]
        score = 0.6 if bias == "BULLISH" else -0.6 if bias == "BEARISH" else 0.0
        if r["harmonic"]["inverted"]:
            score = 0.0
        return {"p_fused": round(score, 4), "confidence": r["bias"]["coherence"],
                "veto_active": r["harmonic"]["inverted"], "active_lambdas": [],
                "regime": "STABLE", "status": "FALLBACK_FUSION"}

    def _mandra(self, r):
        m = self.modules.get("mandra")
        if m:
            try:
                phi = np.array([r["fusion"]["p_fused"]])
                t = m.evaluate_gate(current_phi=phi, stability=r["fusion"]["confidence"],
                                    raw_size=0.02)
                return {"open": t.is_open, "delta_e": t.energy_delta,
                        "size": t.clamped_size, "regime_stable": t.regime_stable,
                        "status": t.status}
            except Exception:
                pass
        p = r["fusion"]["p_fused"]
        e_curr = p ** 2 * r["fusion"]["confidence"]
        de = e_curr - self.prev_energy
        self.prev_energy = e_curr
        return {"open": de >= 0, "delta_e": round(de, 4),
                "size": 0.02 if de >= 0 else 0.0,
                "regime_stable": True, "status": "GATE_OPEN" if de >= 0 else "VETO"}

    def _veto(self, r):
        reasons = []
        if not r["mandra"]["open"]:                          reasons.append("MANDRA:DE<0")
        if r["topology"]["fractured"]:                       reasons.append("TOPO:H1_FRACTURE")
        if r["fusion"]["veto_active"]:                       reasons.append("FUSION:LAMBDA_VETO")
        if r["harmonic"]["inverted"]:                        reasons.append("L3:LIAR_STATE")
        if not r["kl"]["stable"] and r["kl"]["score"] > 1.0: reasons.append("KL:REGIME_FRACTURE")
        if r["fusion"]["confidence"] < 0.2:                 reasons.append("CONF:INSUFFICIENT")
        decision = ("Halt" if reasons else "Reset" if r["amd"]["R_MASTER"] else "Proceed")
        return {"decision": decision, "reasons": reasons, "trade_allowed": decision == "Proceed"}

    def _sensors(self, r):
        vd = r['vol_decay']; ex = r['expansion']; ha = r['harmonic']
        dr = r['dealing_range']; di = r['displacement']; fv = r['fvg']
        ob = r['ob']; kl = r['kl']; tp = r['topology']
        ma = r['mandra']; se = r['session']; mn = r['manipulation']; sw = r['swings']
        return [
            {'id': 's01', 'name': 'PHASE ENTRAP',  'score': vd['ratio'],                     'active': vd['entrapped']},
            {'id': 's02', 'name': 'EXPANSION',      'score': ex['prob'],                      'active': ex['prob'] > 0.5},
            {'id': 's03', 'name': 'HARMONIC L3',    'score': min(1, ha['phase_diff'] / 3.14), 'active': ha['inverted']},
            {'id': 's04', 'name': 'DEAL RANGE',     'score': dr['coherence'],                 'active': True},
            {'id': 's05', 'name': 'PREM/DISC',      'score': 0.9,                             'active': dr['zone'] != 'NEUTRAL'},
            {'id': 's06', 'name': 'DISPLACEMENT',   'score': di['body_ratio'],                'active': di['is_disp']},
            {'id': 's07', 'name': 'FVG DETECT',     'score': min(1, fv['count'] / 5),         'active': fv['active']},
            {'id': 's08', 'name': 'ORDER BLOCK',    'score': min(1, ob['count'] / 5),         'active': ob['active']},
            {'id': 's09', 'name': 'KL DIVERGE',     'score': min(1, kl['score']),             'active': not kl['stable']},
            {'id': 's10', 'name': 'TOPO FRACT',     'score': min(1, tp['h1_score'] / 5),      'active': tp['fractured']},
            {'id': 's11', 'name': 'MANDRA GATE',    'score': 0.9 if ma['open'] else 0.1,      'active': ma['open']},
            {'id': 's12', 'name': 'SESSION L2',     'score': se['score'],                     'active': se['killzone']},
            {'id': 's13', 'name': 'MANIPULATION',   'score': mn['score'] / 100,               'active': mn['active']},
            {'id': 's14', 'name': 'SWING NODES',    'score': min(1, sw['count'] / 10),        'active': sw['count'] > 0},
        ]

    def _blank(self, bar, idx):
        eq = (bar["high"] + bar["low"]) / 2
        blank_dr = {"high": bar["high"], "low": bar["low"], "eq": eq,
                    "zone": "NEUTRAL", "coherence": 0.5, "status": "INIT"}
        blank_s = [{"id": f"s{i:02d}", "name": "--", "score": 0.0, "active": False}
                   for i in range(1, 15)]
        return {
            "bar": bar, "bar_index": idx, "total_bars": len(self.raw_bars),
            "dealing_range": blank_dr,
            "bias":         {"bias": "NEUTRAL", "eq": eq, "zone": "NEUTRAL", "coherence": 0.5, "valid": False},
            "ipda_phase":   {"phase": self.amd_state, "eq": eq, "confidence": 0.5, "valid": False},
            "eq_cross":     {"zone": "NEUTRAL", "cross": False, "direction": "NONE", "confidence": 0.5},
            "session":      {"active": False, "name": "UNKNOWN", "killzone": False, "score": 0.5, "status": "INIT"},
            "swings":       {"count": 0, "nodes": []},
            "fvg":          {"count": 0, "active": False, "recent": []},
            "ob":           {"count": 0, "active": False, "recent": []},
            "vol_profile":  {"zones": 0, "hvn": []},
            "vol_decay":    {"ratio": 0.0, "entrapped": False, "energy": 0.0, "stasis": 0, "status": "INIT"},
            "displacement": {"is_disp": False, "dir": 0, "body_ratio": 0.0, "range_mult": 0.0, "vetoed": False, "status": "INIT"},
            "harmonic":     {"phase_diff": 0.0, "inverted": False, "freq": 0.0, "trap": "NONE", "status": "INIT"},
            "expansion":    {"sigma": 0, "prob": 0.0, "entrapped": False, "target": 0.0, "status": "INIT"},
            "manipulation": {"active": False, "score": 0, "level": "NONE", "wick": 0.0, "status": "INIT"},
            "kl":           {"score": 0.0, "stable": True, "h_curr": 0.0, "h_ref": 0.0, "status": "INIT"},
            "topology":     {"h1_score": 0.0, "fractured": False, "islands": 0, "status": "COMPACT_CLOUD"},
            "amd":          {"state": self.amd_state, "prev": self.amd_state, "changed": False, "R_MASTER": False},
            "fusion":       {"p_fused": 0.0, "confidence": 0.5, "veto_active": False, "active_lambdas": [], "regime": "INIT", "status": "INIT"},
            "mandra":       {"open": True, "delta_e": 0.0, "size": 0.0, "regime_stable": True, "status": "INIT"},
            "veto":         {"decision": "Proceed", "reasons": [], "trade_allowed": True},
            "sensors":      blank_s,
        }


def _imp(module_path, class_name):
    import importlib
    return getattr(importlib.import_module(module_path), class_name)


def _to_df(bars):
    df = pd.DataFrame(bars)
    if "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("datetime")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean().fillna(
                 (df["high"] - df["low"]).mean())
    df["atr20"] = (df["high"] - df["low"]).rolling(20).mean().fillna(
                   (df["high"] - df["low"]).mean())
    return df

# Plugin system — lazy load to avoid circular imports
_plugin_mgr = None
_plugin_mgr_failed = False
def _get_plugins():
    global _plugin_mgr, _plugin_mgr_failed
    if _plugin_mgr_failed:
        return None
    if _plugin_mgr is None:
        try:
            import sys as _sys
            _bd = os.path.dirname(os.path.abspath(__file__))
            if _bd not in _sys.path:
                _sys.path.insert(0, _bd)
            from plugin_manager import get_plugin_manager
            _plugin_mgr = get_plugin_manager()
        except Exception as e:
            print(f"[SMK] Plugin manager not available: {e}")
            _plugin_mgr_failed = True
            _plugin_mgr = None
    return _plugin_mgr

_bridge = None
def _get_bridge():
    global _bridge
    if _bridge is None:
        try:
            from aegis_bridge import get_bridge
            _bridge = get_bridge()
        except Exception:
            pass
    return _bridge

def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    elif isinstance(obj, (np.generic, np.ndarray)):
        return obj.item() if hasattr(obj, 'item') else obj.tolist()
    elif isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    return str(obj)
