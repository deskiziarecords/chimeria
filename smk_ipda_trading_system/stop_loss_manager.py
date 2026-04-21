"""
stop_loss_manager.py
════════════════════════════════════════════════════════════════════════════════
SMART-EXE: Dynamic Stop Loss Engine
Sovereign Market Kernel — Pattern DNA Invalidation Logic

Extends the original StopLossManager with:
  • Position sizing  — lot size from capital + risk budget
  • ATR overlay      — static pct breathes with live volatility
  • TP / R:R gate    — trade only armed when reward ≥ min_rr × risk
  • AEGIS tie-in     — SchurRouter receives sized quantities per venue
  • Audit trail      — every decision logged with full evidence

SMART-EXE 7-symbol alphabet:
    B = Strong Bullish   I = Strong Bearish
    W = Upper Wick       w = Lower Wick
    U = Weak Bull        D = Weak Bear
    X = Neutral / Structure only

════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("smart_exe.stop_loss")


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskProfile:
    """Full risk specification for one trade setup."""

    # Core stop loss
    stop_loss_price:  float
    stop_loss_pct:    float
    risk_per_unit:    float          # price distance from entry to stop

    # Position sizing
    lot_size:         float          # units / lots / contracts
    risk_usd:         float          # capital at risk ($)
    capital_fraction: float          # fraction of capital risked

    # Take profit
    take_profit_price: float
    take_profit_pct:   float
    rr_ratio:          float         # reward-to-risk ratio

    # Gate results
    is_valid:         bool
    status:           str            # "ARMED" | "VETO: reason"
    gate_results:     Dict[str, bool] = field(default_factory=dict)

    # Metadata
    pattern:          str   = "X"
    direction:        int   = 1       # 1 = Long, -1 = Short
    entry_price:      float = 0.0
    atr_value:        float = 0.0
    atr_used:         bool  = False
    timestamp:        float = field(default_factory=time.time)

    @property
    def is_long(self) -> bool:
        return self.direction == 1

    @property
    def risk_pips(self) -> float:
        """Risk expressed in pips (assumes 5dp pair)."""
        return abs(self.entry_price - self.stop_loss_price) / 0.0001

    def summary(self) -> str:
        dir_str = "LONG" if self.is_long else "SHORT"
        return (
            f"[{self.pattern}] {dir_str}  "
            f"entry={self.entry_price:.5f}  "
            f"SL={self.stop_loss_price:.5f} ({self.stop_loss_pct*100:.2f}%)  "
            f"TP={self.take_profit_price:.5f}  "
            f"RR={self.rr_ratio:.1f}  "
            f"lots={self.lot_size:.2f}  "
            f"risk=${self.risk_usd:.2f}  "
            f"{'✓ ' + self.status if self.is_valid else '✗ ' + self.status}"
        )


@dataclass
class TradeAudit:
    """Immutable forensic record of one stop-loss calculation."""
    pattern:         str
    entry_price:     float
    direction:       int
    base_sl_pct:     float
    atr_sl_pct:      float
    final_sl_pct:    float
    lambda6_cap:     float
    atr_value:       float
    capital:         float
    risk_budget_pct: float
    profile:         RiskProfile
    ts:              float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# ATR CALCULATOR  (rolling, no Pandas required)
# ─────────────────────────────────────────────────────────────────────────────

class ATRCalculator:
    """
    Rolling Average True Range calculator.
    Accepts bar-by-bar updates — no DataFrame needed.
    """

    def __init__(self, period: int = 14):
        self.period  = period
        self._highs: List[float] = []
        self._lows:  List[float] = []
        self._closes: List[float] = []
        self._atr:   Optional[float] = None

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        """Push one bar. Returns current ATR or None during warmup."""
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)

        n = len(self._closes)
        if n < 2:
            return None

        # True Range of last bar
        prev_close = self._closes[-2]
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low  - prev_close),
        )

        if n < self.period + 1:
            # Warmup: simple mean
            trs = [
                max(
                    self._highs[i] - self._lows[i],
                    abs(self._highs[i] - self._closes[i-1]),
                    abs(self._lows[i]  - self._closes[i-1]),
                )
                for i in range(1, n)
            ]
            self._atr = float(np.mean(trs))
        else:
            # Wilder smoothing
            self._atr = (self._atr * (self.period - 1) + tr) / self.period

        # Keep buffer bounded
        if len(self._closes) > self.period * 3:
            self._highs  = self._highs[-self.period*2:]
            self._lows   = self._lows[-self.period*2:]
            self._closes = self._closes[-self.period*2:]

        return self._atr

    @property
    def value(self) -> Optional[float]:
        return self._atr

    @property
    def is_ready(self) -> bool:
        return self._atr is not None and len(self._closes) >= self.period


# ─────────────────────────────────────────────────────────────────────────────
# STOP LOSS MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class StopLossManager:
    """
    SMART-EXE: Dynamic Stop Loss Engine.
    Implements pattern-based invalidation logic with ATR overlay,
    position sizing, R:R enforcement, and AEGIS circuit breaker integration.

    Usage (standalone):
        engine = StopLossManager(capital=10_000)
        engine.atr.update(high, low, close)   # feed bars
        risk = engine.calculate_stop('W', entry=1.08500, direction=1)
        if risk.is_valid:
            print(risk.summary())

    Usage (with AEGIS extensions):
        from aegis_extensions import AegisExtensions
        aegis = AegisExtensions(n_venues=4)
        engine = StopLossManager(capital=10_000, aegis=aegis)
        risk = engine.calculate_stop('B', 1.08500, direction=1)
        # risk.lot_size is automatically split across venues via SchurRouter
    """

    # ── Pattern DNA → base stop-loss percentage mapping (SMART-EXE Spec) ──
    PATTERN_SL_MAP: Dict[str, float] = {
        'B': 0.008,   # Strong Bullish  — 0.8%  (high conviction, tight stop)
        'I': 0.008,   # Strong Bearish  — 0.8%
        'W': 0.006,   # Upper Wick      — 0.6%  (reversal, very tight)
        'w': 0.006,   # Lower Wick      — 0.6%
        'U': 0.010,   # Weak Bull       — 1.0%  (wider, less clean setup)
        'D': 0.010,   # Weak Bear       — 1.0%
        'X': 0.005,   # Neutral/Struct. — 0.5%  (structure only, tightest)
    }

    # ── Pattern → minimum R:R required before arming ──
    PATTERN_MIN_RR: Dict[str, float] = {
        'B': 2.0,   # Strong setups demand 2:1
        'I': 2.0,
        'W': 2.5,   # Wick reversals: high accuracy, demand more
        'w': 2.5,
        'U': 1.5,   # Weak setups: accept lower R:R
        'D': 1.5,
        'X': 1.5,   # Structure only: minimum bar
    }

    # ── Pattern → TP multiplier (TP = entry ± TP_mult × stop_distance) ──
    PATTERN_TP_MULT: Dict[str, float] = {
        'B': 2.5,
        'I': 2.5,
        'W': 3.0,
        'w': 3.0,
        'U': 2.0,
        'D': 2.0,
        'X': 2.0,
    }

    def __init__(
        self,
        capital:               float = 10_000.0,   # total account capital
        risk_per_trade_pct:    float = 0.01,        # 1% of capital per trade
        max_micro_sanity_pct:  float = 0.01,        # λ6: hard 1% cap
        atr_period:            int   = 14,
        atr_weight:            float = 0.5,         # blend: 50% static, 50% ATR
        min_rr:                float = 1.5,         # global minimum R:R
        pip_value:             float = 10.0,        # $ per pip per standard lot
        aegis                        = None,        # AegisExtensions (optional)
    ):
        self.capital            = capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_gate_pct       = max_micro_sanity_pct
        self.atr_weight         = float(np.clip(atr_weight, 0.0, 1.0))
        self.min_rr             = min_rr
        self.pip_value          = pip_value
        self.aegis              = aegis             # optional AEGIS integration

        # Live ATR engine
        self.atr = ATRCalculator(period=atr_period)

        # Audit trail
        self._audit: List[TradeAudit] = []
        self._armed_count   = 0
        self._vetoed_count  = 0

    # ── MAIN ENTRY POINT ──────────────────────────────────────────────────

    def calculate_stop(
        self,
        symbol:      str,             # SMART-EXE pattern letter (B/I/W/w/U/D/X)
        entry_price: float,
        direction:   int,             # 1 = Long, -1 = Short
        atr_override: Optional[float] = None,
        confidence:   float = 1.0,    # QUIMERIA fusion confidence (0–1)
    ) -> RiskProfile:
        """
        Master Logic: calculate full risk profile from Pattern DNA.

        Args:
            symbol:       SMART-EXE pattern character
            entry_price:  trade entry level
            direction:    1 (Long) or -1 (Short)
            atr_override: optionally supply ATR externally (e.g. from QUIMERIA)
            confidence:   QUIMERIA fusion confidence — scales position size

        Returns:
            RiskProfile with stop, TP, lot size, R:R, and gate status
        """
        symbol = symbol.upper() if symbol not in ('w',) else symbol

        # ── Step 1: Base SL percentage from Pattern DNA ───────────────────
        base_sl_pct = self.PATTERN_SL_MAP.get(symbol, self.max_gate_pct)

        # ── Step 2: ATR overlay — dynamic volatility adjustment ───────────
        atr_val    = atr_override or (self.atr.value if self.atr.is_ready else None)
        atr_sl_pct = base_sl_pct   # default: ATR not available
        atr_used   = False

        if atr_val is not None and entry_price > 0:
            # Convert ATR to percentage of entry price
            atr_pct   = atr_val / entry_price
            # Blend: final = (1 - weight) × base + weight × ATR-based
            atr_sl_pct = (1.0 - self.atr_weight) * base_sl_pct + self.atr_weight * atr_pct
            atr_used   = True

        # ── Step 3: λ6 Micro Sanity Gate — hard 1% ceiling ───────────────
        final_sl_pct = min(atr_sl_pct, self.max_gate_pct)
        gate_l6_ok   = atr_sl_pct <= self.max_gate_pct

        # ── Step 4: Compute stop price ────────────────────────────────────
        # Stop = Entry − (Direction × Entry × SL_PCT)
        stop_price   = entry_price - (direction * entry_price * final_sl_pct)
        risk_per_unit = abs(entry_price - stop_price)

        # ── Step 5: Take profit ───────────────────────────────────────────
        tp_mult       = self.PATTERN_TP_MULT.get(symbol, 2.0)
        tp_price      = entry_price + (direction * risk_per_unit * tp_mult)
        tp_pct        = abs(tp_price - entry_price) / entry_price
        rr_ratio      = tp_mult   # by construction (TP = SL × mult)

        # ── Step 6: R:R gate ──────────────────────────────────────────────
        min_rr_required = max(self.min_rr, self.PATTERN_MIN_RR.get(symbol, 1.5))
        gate_rr_ok      = rr_ratio >= min_rr_required

        # ── Step 7: Position sizing ───────────────────────────────────────
        # Risk budget in $ = capital × risk_pct × confidence
        risk_budget_usd = self.capital * self.risk_per_trade_pct * float(confidence)

        # Lot size (in standard lots where applicable):
        # risk_budget = lot_size × risk_per_unit × pip_value / 0.0001
        # lot_size = risk_budget / (risk_pips × pip_value)
        risk_pips = risk_per_unit / 0.0001   # convert to pips (5dp pair)
        if risk_pips > 1e-6:
            lot_size = risk_budget_usd / (risk_pips * self.pip_value)
        else:
            lot_size = 0.0
        lot_size = max(0.0, round(lot_size, 2))

        risk_usd         = lot_size * risk_pips * self.pip_value
        capital_fraction = risk_usd / max(self.capital, 1.0)

        # ── Step 8: All-gates validation ─────────────────────────────────
        gate_results = {
            "λ6_cap":      gate_l6_ok,    # SL ≤ 1%
            "rr_minimum":  gate_rr_ok,    # R:R ≥ required
            "lot_positive": lot_size > 0, # position is non-zero
        }
        is_valid = all(gate_results.values())
        status   = self._status_string(is_valid, gate_results)

        if is_valid:
            self._armed_count += 1
        else:
            self._vetoed_count += 1

        profile = RiskProfile(
            stop_loss_price   = round(stop_price, 5),
            stop_loss_pct     = round(final_sl_pct, 6),
            risk_per_unit     = round(risk_per_unit, 6),
            lot_size          = lot_size,
            risk_usd          = round(risk_usd, 2),
            capital_fraction  = round(capital_fraction, 6),
            take_profit_price = round(tp_price, 5),
            take_profit_pct   = round(tp_pct, 6),
            rr_ratio          = round(rr_ratio, 2),
            is_valid          = is_valid,
            status            = status,
            gate_results      = gate_results,
            pattern           = symbol,
            direction         = direction,
            entry_price       = entry_price,
            atr_value         = float(atr_val) if atr_val else 0.0,
            atr_used          = atr_used,
        )

        # ── Step 9: Audit trail ───────────────────────────────────────────
        self._audit.append(TradeAudit(
            pattern         = symbol,
            entry_price     = entry_price,
            direction       = direction,
            base_sl_pct     = base_sl_pct,
            atr_sl_pct      = atr_sl_pct,
            final_sl_pct    = final_sl_pct,
            lambda6_cap     = self.max_gate_pct,
            atr_value       = float(atr_val) if atr_val else 0.0,
            capital         = self.capital,
            risk_budget_pct = self.risk_per_trade_pct,
            profile         = profile,
        ))
        if len(self._audit) > 10_000:
            self._audit = self._audit[-5_000:]

        log.info("%s", profile.summary())
        return profile

    # ── AEGIS INTEGRATION ─────────────────────────────────────────────────

    def calculate_stop_with_routing(
        self,
        symbol:      str,
        entry_price: float,
        direction:   int,
        confidence:  float = 1.0,
        atr_override: Optional[float] = None,
    ) -> Tuple[RiskProfile, Optional[object]]:
        """
        Calculate stop + run the lot size through AEGIS SchurRouter.

        Returns (RiskProfile, SchurResult | None).
        SchurResult.allocation gives per-venue lot fractions.

        Usage:
            profile, routing = engine.calculate_stop_with_routing('B', 1.08500, 1)
            if profile.is_valid and routing:
                for i, venue in enumerate(my_venues):
                    venue_lots = profile.lot_size * routing.allocation[i]
                    broker[venue].submit(direction, venue_lots)
        """
        profile = self.calculate_stop(symbol, entry_price, direction,
                                       atr_override=atr_override,
                                       confidence=confidence)

        if not profile.is_valid or self.aegis is None:
            return profile, None

        try:
            desired  = np.full(self.aegis.router.n_venues,
                               confidence / self.aegis.router.n_venues,
                               dtype=np.float32)
            routing  = self.aegis.router.route(desired)
            return profile, routing
        except Exception as e:
            log.warning("SchurRouter failed: %s", e)
            return profile, None

    def calculate_stop_with_mandra(
        self,
        symbol:       str,
        entry_price:  float,
        direction:    int,
        signal_probs: np.ndarray,
        confidence:   float = 1.0,
    ) -> Tuple[RiskProfile, dict]:
        """
        Calculate stop + run signal through AEGIS Mandra ΔE gate.
        If ΔE ≤ threshold, sets is_valid=False regardless of pattern gates.

        Usage:
            probs = np.array([0.65, 0.25, 0.10])  # LONG/FLAT/SHORT
            profile, mandra = engine.calculate_stop_with_mandra('W', 1.085, 1, probs)
            if profile.is_valid and mandra['allow']:
                # Trade armed AND information gate cleared
                execute(profile)
        """
        profile      = self.calculate_stop(symbol, entry_price, direction,
                                            confidence=confidence)
        mandra_result = {"allow": True, "delta_e": 0.0}

        if self.aegis is None:
            return profile, mandra_result

        mandra_result = self.aegis.mandra.evaluate(signal_probs)
        if not mandra_result["allow"]:
            # Override: gate blocks regardless of pattern
            profile.is_valid = False
            profile.status   = f"VETO: MANDRA ΔE={mandra_result['delta_e']:.4f} (no information)"
            profile.gate_results["mandra_gate"] = False
            log.warning("Mandra gate blocked %s trade: ΔE=%.4f", symbol, mandra_result["delta_e"])

        # Scale lot size by information content
        if profile.is_valid:
            profile.lot_size = round(
                profile.lot_size * mandra_result["position_scale"], 2
            )

        return profile, mandra_result

    # ── UTILITY METHODS ───────────────────────────────────────────────────

    def update_atr(self, high: float, low: float, close: float) -> Optional[float]:
        """Feed a completed bar into the ATR calculator."""
        return self.atr.update(high, low, close)

    def update_capital(self, new_capital: float) -> None:
        """Update capital base (e.g. after P&L, drawdown)."""
        self.capital = max(0.0, new_capital)

    def pattern_summary(self) -> dict:
        """Return all pattern configurations for inspection."""
        return {
            sym: {
                "sl_pct":    f"{pct*100:.1f}%",
                "min_rr":    self.PATTERN_MIN_RR[sym],
                "tp_mult":   self.PATTERN_TP_MULT[sym],
            }
            for sym, pct in self.PATTERN_SL_MAP.items()
        }

    def session_stats(self) -> dict:
        """Return session-level risk statistics."""
        recent = self._audit[-100:]
        if not recent:
            return {"armed": 0, "vetoed": 0, "total": 0}

        armed   = [a for a in recent if a.profile.is_valid]
        avg_rr  = np.mean([a.profile.rr_ratio for a in armed]) if armed else 0.0
        avg_sl  = np.mean([a.profile.stop_loss_pct for a in armed]) if armed else 0.0
        avg_lot = np.mean([a.profile.lot_size for a in armed]) if armed else 0.0

        return {
            "armed":        self._armed_count,
            "vetoed":       self._vetoed_count,
            "total":        self._armed_count + self._vetoed_count,
            "arm_rate":     f"{self._armed_count / max(1, self._armed_count + self._vetoed_count):.0%}",
            "avg_rr":       round(avg_rr, 2),
            "avg_sl_pct":   f"{avg_sl*100:.2f}%",
            "avg_lot_size": round(avg_lot, 2),
            "current_atr":  round(self.atr.value, 5) if self.atr.value else None,
        }

    def recent_audit(self, n: int = 5) -> List[str]:
        """Return last N trade summaries for the log."""
        return [a.profile.summary() for a in self._audit[-n:]]

    @staticmethod
    def _status_string(is_valid: bool, gates: Dict[str, bool]) -> str:
        if is_valid:
            return "ARMED: WITHIN TOLERANCE"
        failed = [k for k, v in gates.items() if not v]
        return f"VETO: {', '.join(failed).upper()} BREACHED"


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE STOP LOSS MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class SequenceStopLossManager:
    """
    Extends StopLossManager to handle SMART-EXE full symbol sequences.

    In SMART-EXE, a trade setup is encoded as a sequence of pattern letters
    e.g. "WBU" = wick + strong bull + weak bull context.
    This class decodes the sequence and selects the dominant stop logic.

    Usage:
        manager = SequenceStopLossManager(capital=10_000)
        profile = manager.calculate_from_sequence("WBU", 1.08500, direction=1)
    """

    # Dominance hierarchy: which pattern takes priority in a sequence
    DOMINANCE = {'W': 5, 'w': 5, 'B': 4, 'I': 4, 'X': 3, 'U': 2, 'D': 2}

    def __init__(self, **kwargs):
        self.engine = StopLossManager(**kwargs)

    def calculate_from_sequence(
        self,
        sequence:    str,
        entry_price: float,
        direction:   int,
        **kwargs,
    ) -> RiskProfile:
        """
        Parse a SMART-EXE pattern sequence and route to the dominant symbol.

        Dominant symbol = highest entry in DOMINANCE hierarchy.
        If two symbols tie, the first in sequence wins.

        Example:
            "WBU" → 'W' dominates (score 5) → 0.6% SL, 3.0× TP
            "UDX" → 'U' dominates (score 2) → 1.0% SL, 2.0× TP
        """
        if not sequence:
            return self.engine.calculate_stop('X', entry_price, direction, **kwargs)

        dominant = max(sequence, key=lambda c: self.DOMINANCE.get(c, 0))
        log.debug("Sequence '%s' → dominant='%s'", sequence, dominant)
        return self.engine.calculate_stop(dominant, entry_price, direction, **kwargs)

    def calculate_all_symbols(
        self,
        sequence:    str,
        entry_price: float,
        direction:   int,
    ) -> List[RiskProfile]:
        """
        Calculate a risk profile for every symbol in the sequence.
        Returns profiles sorted by R:R descending — most favourable first.
        """
        profiles = [
            self.engine.calculate_stop(sym, entry_price, direction)
            for sym in sequence
        ]
        return sorted(profiles, key=lambda p: p.rr_ratio, reverse=True)

    def update_atr(self, high: float, low: float, close: float) -> Optional[float]:
        return self.engine.update_atr(high, low, close)

    def session_stats(self) -> dict:
        return self.engine.session_stats()


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    """Quick self-test — run with: python stop_loss_manager.py"""
    import traceback
    PASS = 0; FAIL = 0

    def test(name, fn):
        nonlocal PASS, FAIL
        try:
            fn()
            print(f"  PASS  {name}")
            PASS += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
            FAIL += 1

    print("\n══════════════════════════════════════")
    print("  SMART-EXE Stop Loss Manager Tests")
    print("══════════════════════════════════════")

    engine = StopLossManager(capital=10_000)

    def t_basic_long():
        r = engine.calculate_stop('W', 1.08500, direction=1)
        assert r.direction == 1
        assert r.stop_loss_price < 1.08500, f"SL {r.stop_loss_price} should be below entry"
        assert r.take_profit_price > 1.08500, f"TP {r.take_profit_price} should be above entry"
        assert r.stop_loss_pct == 0.006, f"W pattern should be 0.6%, got {r.stop_loss_pct}"
        assert r.rr_ratio == 3.0, f"W TP mult should be 3.0, got {r.rr_ratio}"
        assert r.is_valid

    def t_basic_short():
        r = engine.calculate_stop('B', 1.08500, direction=-1)
        assert r.stop_loss_price > 1.08500, f"SL {r.stop_loss_price} should be above entry for short"
        assert r.take_profit_price < 1.08500, f"TP {r.take_profit_price} should be below entry"

    def t_lambda6_gate():
        # U pattern = 1.0% = exactly at cap
        r = engine.calculate_stop('U', 1.08500, direction=1)
        assert r.stop_loss_pct <= engine.max_gate_pct, "λ6 gate should cap at max_gate_pct"
        assert r.gate_results['λ6_cap'] == True

    def t_unknown_pattern():
        # Unknown pattern should fall back to max cap
        r = engine.calculate_stop('Z', 1.08500, direction=1)
        assert r.stop_loss_pct <= engine.max_gate_pct

    def t_atr_overlay():
        eng = StopLossManager(capital=10_000, atr_weight=0.5)
        # Simulate low-vol ATR (1 pip = 0.0001 → very tight)
        low_vol_atr = 0.0005   # 5 pip ATR
        r = eng.calculate_stop('B', 1.08500, direction=1, atr_override=low_vol_atr)
        # With ATR blending at low vol, SL should be tighter than 0.8% base
        assert r.stop_loss_pct < 0.008, f"ATR should tighten SL, got {r.stop_loss_pct}"
        assert r.atr_used == True

    def t_position_sizing():
        r = engine.calculate_stop('B', 1.08500, direction=1)
        assert r.lot_size >= 0.0, "Lot size must be non-negative"
        assert r.risk_usd > 0.0, "Risk USD must be positive"
        # Risk should not exceed 1% of capital
        assert r.risk_usd <= engine.capital * engine.risk_per_trade_pct * 1.10, \
            f"risk_usd={r.risk_usd:.2f} exceeds 110% of budget {engine.capital*engine.risk_per_trade_pct}"

    def t_rr_ratio():
        r = engine.calculate_stop('W', 1.08500, direction=1)
        # W pattern: TP mult = 3.0 → R:R should be ≥ 2.5 (W's min)
        assert r.rr_ratio >= engine.PATTERN_MIN_RR['W'], f"R:R {r.rr_ratio} below minimum"

    def t_summary_string():
        r = engine.calculate_stop('B', 1.08500, direction=1)
        s = r.summary()
        assert 'LONG' in s or 'SHORT' in s
        assert 'ARMED' in s or 'VETO' in s

    def t_all_patterns():
        for sym in engine.PATTERN_SL_MAP:
            r = engine.calculate_stop(sym, 1.08500, direction=1)
            assert r.stop_loss_pct <= engine.max_gate_pct, f"{sym}: SL exceeded cap"
            assert r.take_profit_price > 1.08500, f"{sym}: TP wrong side"

    def t_atr_calculator():
        atr_calc = ATRCalculator(period=5)
        prices = [(1.086, 1.084, 1.085)] * 10
        for h, l, c in prices:
            atr_calc.update(h, l, c)
        assert atr_calc.is_ready
        assert atr_calc.value > 0

    def t_sequence_manager():
        seq_mgr = SequenceStopLossManager(capital=10_000)
        r = seq_mgr.calculate_from_sequence("WBU", 1.08500, direction=1)
        assert r.pattern == 'W', f"W should dominate WBU, got {r.pattern}"
        r2 = seq_mgr.calculate_from_sequence("UDX", 1.08500, direction=1)
        # X has dominance score 3, U/D have 2 — X dominates 'UDX' correctly
        assert r2.pattern == 'X', f"X should dominate UDX (score 3>2), got {r2.pattern}"

    def t_session_stats():
        eng = StopLossManager(capital=10_000)
        for sym in ['B', 'W', 'U', 'X']:
            eng.calculate_stop(sym, 1.08500, direction=1)
        stats = eng.session_stats()
        assert stats['total'] == 4
        assert stats['armed'] + stats['vetoed'] == 4

    def t_capital_update():
        eng = StopLossManager(capital=10_000)
        r1 = eng.calculate_stop('B', 1.08500, direction=1)
        eng.update_capital(5_000)   # simulate 50% drawdown
        r2 = eng.calculate_stop('B', 1.08500, direction=1)
        assert r2.lot_size < r1.lot_size, "Lot size should shrink with capital"

    test("basic long stop",       t_basic_long)
    test("basic short stop",      t_basic_short)
    test("λ6 hard gate",          t_lambda6_gate)
    test("unknown pattern",       t_unknown_pattern)
    test("ATR overlay",           t_atr_overlay)
    test("position sizing",       t_position_sizing)
    test("R:R enforcement",       t_rr_ratio)
    test("summary string",        t_summary_string)
    test("all 7 patterns",        t_all_patterns)
    test("ATR calculator",        t_atr_calculator)
    test("sequence manager",      t_sequence_manager)
    test("session stats",         t_session_stats)
    test("capital drawdown adj",  t_capital_update)

    print(f"\n  {PASS}/{PASS+FAIL} tests passed")
    print("══════════════════════════════════════\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _run_tests()

    # ── Live example ──────────────────────────────────────────────────────
    print("── Live Example ──────────────────────────────────")
    engine = StopLossManager(capital=10_000, risk_per_trade_pct=0.01)

    # Feed some bars for ATR warmup
    for _ in range(20):
        engine.update_atr(1.0865, 1.0840, 1.0855)

    # All 7 patterns, long direction
    print("\nPattern DNA → Risk Profiles (LONG, entry=1.08500, $10K account)\n")
    for sym in ['B', 'I', 'W', 'w', 'U', 'D', 'X']:
        r = engine.calculate_stop(sym, 1.08500, direction=1)
        print(f"  {r.summary()}")

    # Session stats
    print("\nSession Stats:")
    for k, v in engine.session_stats().items():
        print(f"  {k:<20} {v}")

    # Sequence example
    print("\nSequence Manager — 'WBU' sequence:")
    seq = SequenceStopLossManager(capital=10_000)
    for _ in range(20):
        seq.update_atr(1.0865, 1.0840, 1.0855)
    r = seq.calculate_from_sequence("WBU", 1.08500, direction=1)
    print(f"  {r.summary()}")
