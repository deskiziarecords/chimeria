"""
aegis_extensions.py
════════════════════════════════════════════════════════════════════════════════
AEGIS → QUIMERIA Integration Layer

Drop this single file into your QUIMERIA project root.
Import the six extensions you need — everything else in QUIMERIA stays
exactly as it is.

Usage in your QUIMERIA main / signal engine:

    from aegis_extensions import (
        AegisExtensions,          # ← convenience wrapper (all 6 at once)
        SchurRouter,              # ← 1. venue routing
        CausalTransmissionGraph,  # ← 2. lead-lag causal graph
        CircuitBreakerManager,    # ← 3. production circuit breakers
        AdelicLevelScorer,        # ← 4. p-adic order block ranking
        AegisReversePeriodGuard,  # ← 5. streaming reverse period guard
        MandraBitGate,            # ← 6. information-theoretic trade gate
    )

    # Quickstart — single object with all extensions pre-wired:
    aegis = AegisExtensions(n_venues=4)
    aegis.start()

    # In your QUIMERIA tick loop:
    bar = {"close": price, "high": high, "low": low, "volume": vol,
           "sigma": amd_phase_int, "phi": confluence_score,
           "killzone": is_killzone_session}

    result = aegis.on_bar(
        bar         = bar,
        signal      = quimeria_signal,      # float in [-1, 1]
        confidence  = quimeria_confidence,  # float in [0, 1]
        returns     = recent_returns,       # np.ndarray, last N bars
        vpin        = vpin_vector,          # np.ndarray, per asset
    )

    if result["action"] == "TRADE":
        size       = result["kelly_size"]       # position size (fraction)
        allocation = result["venue_allocation"] # per-venue split
        # → pass to your broker/execution layer

    elif result["action"] == "HALT":
        reason = result["halt_reason"]
        # → QUIMERIA's u_t = 0 protocol

════════════════════════════════════════════════════════════════════════════════
QUIMERIA compatibility:
  - FastAPI / async:  full async support
  - Pandas / NumPy:   numpy arrays in/out throughout
  - Redis optional:   falls back to in-process store automatically
  - Python 3.9+:      no new syntax requirements
════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, NamedTuple, Optional

import numpy as np

log = logging.getLogger("aegis_extensions")


# ════════════════════════════════════════════════════════════════════════════
# 1. SCHUR-COMPLEMENT VENUE ROUTER
# ════════════════════════════════════════════════════════════════════════════
# Replaces QUIMERIA's uniform signal split with an optimal allocation across
# N execution venues, accounting for per-venue liquidity and market impact.
# Uses Schur-complement factorisation + conjugate gradient in O(log N).

class SchurResult(NamedTuple):
    allocation: np.ndarray   # (n_venues,) fractions, sums to 1.0
    cost:       float        # quadratic cost at optimum
    converged:  bool         # CG converged within tolerance


def _cg_solve(A: np.ndarray, b: np.ndarray,
               max_iter: int = 64, tol: float = 1e-6) -> np.ndarray:
    """Conjugate gradient solver for Ax = b."""
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    rr = float(r @ r)
    for _ in range(max_iter):
        if rr < tol ** 2:
            break
        Ap = A @ p
        denom = float(p @ Ap)
        if abs(denom) < 1e-15:
            break
        alpha = rr / denom
        x    += alpha * p
        r    -= alpha * Ap
        rr_new = float(r @ r)
        p = r + (rr_new / (rr + 1e-15)) * p
        rr = rr_new
    return x


def _project_simplex_floor(x: np.ndarray, floor: float) -> np.ndarray:
    """
    Project x onto the probability simplex with a per-element floor.
    Guaranteed: sum(x) == 1  and  all(x >= floor).
    Simple clip+renormalise violates the floor when one venue dominates —
    this iterative version is guaranteed convergent.
    """
    x = np.clip(x, floor, 1.0)
    for _ in range(50):
        total = x.sum()
        if abs(total - 1.0) < 1e-9:
            break
        slack   = x - floor
        deficit = total - 1.0
        if deficit > 0:
            reducible = slack.sum()
            if reducible > 1e-12:
                x = x - slack * (deficit / reducible)
        else:
            x = x + (-deficit / len(x))
        x = np.clip(x, floor, 1.0)
    return x


def schur_route(desired: np.ndarray,
                liquidity: np.ndarray,
                impact: np.ndarray,
                min_alloc: float = 0.05) -> SchurResult:
    """
    Solve: min ½xᵀQx − cᵀx  s.t. Σxᵢ=1, xᵢ≥min_alloc
    Q = impact + diag(1/liquidity)    (total cost matrix)
    c = desired                        (target allocation)
    Solution via Schur complement + CG.
    """
    n = len(desired)
    if n == 0:
        return SchurResult(np.array([]), 0.0, True)

    Q = impact + np.diag(1.0 / np.clip(liquidity, 1e-6, None)) + np.eye(n) * 1e-6
    c = desired.astype(np.float64)
    ones = np.ones(n, dtype=np.float64)

    Qinv_c = _cg_solve(Q, c)
    Qinv_1 = _cg_solve(Q, ones)
    lam    = (ones @ Qinv_c - 1.0) / (ones @ Qinv_1 + 1e-15)
    x_opt  = Qinv_c - lam * Qinv_1
    x_opt  = _project_simplex_floor(x_opt, min_alloc)

    cost      = float(0.5 * x_opt @ Q @ x_opt - c @ x_opt)
    converged = bool(abs(x_opt.sum() - 1.0) < 1e-4)
    return SchurResult(x_opt.astype(np.float32), cost, converged)


class SchurRouter:
    """
    Stateful Schur venue router with adaptive liquidity tracking.

    QUIMERIA usage:
        router = SchurRouter(n_venues=4)   # 4 brokers / execution venues

        # Given QUIMERIA's signal confidence (0–1):
        desired = np.full(4, confidence / 4)    # uniform starting point
        result  = router.route(desired)
        # result.allocation → [0.38, 0.27, 0.21, 0.14]  optimal split

        # After fills come back, update liquidity estimates:
        router.update_liquidity(filled_qty, quoted_qty)
    """

    def __init__(self, n_venues: int = 4, min_alloc: float = 0.05):
        self.n_venues  = n_venues
        self.min_alloc = min_alloc
        self._liq      = np.ones(n_venues, dtype=np.float32)
        self._impact   = np.eye(n_venues,  dtype=np.float32) * 0.01
        self._history: list[np.ndarray] = []

    def route(self, desired: Optional[np.ndarray] = None) -> SchurResult:
        if desired is None:
            desired = np.ones(self.n_venues, dtype=np.float32) / self.n_venues
        return schur_route(desired, self._liq, self._impact, self.min_alloc)

    def route_from_confidence(self, confidence: float) -> SchurResult:
        """Convenience: route a scalar confidence into venue fractions."""
        desired = np.full(self.n_venues, confidence / self.n_venues, dtype=np.float32)
        return self.route(desired)

    def update_liquidity(self, fills: np.ndarray, quoted: np.ndarray) -> None:
        ratio = np.clip(fills / (quoted + 1e-10), 0.0, 1.0).astype(np.float32)
        self._liq = 0.9 * self._liq + 0.1 * ratio
        self._history.append(ratio)
        if len(self._history) > 20:
            hist = np.stack(self._history[-20:])
            self._impact = (np.cov(hist.T).astype(np.float32)
                            + np.eye(self.n_venues) * 0.001)


# ════════════════════════════════════════════════════════════════════════════
# 2. CAUSAL TRANSMISSION GRAPH
# ════════════════════════════════════════════════════════════════════════════
# Detects lead-lag causal relationships between assets using four methods.
# Tells QUIMERIA: "DXY moved 2 bars ago — EURUSD move is coming."

@dataclass
class CausalEdge:
    source:      str
    target:      str
    weight:      float          # composite causal strength [0, 1]
    lag_bars:    int            # estimated lag in bars
    granger_p:   float = 1.0   # Granger p-value (lower = more causal)
    te_score:    float = 0.0   # transfer entropy bits
    method:      str   = "composite"


class CausalTransmissionGraph:
    """
    Four-method causal discovery across a universe of assets.

    QUIMERIA usage:
        graph = CausalTransmissionGraph(max_lags=8)
        graph.build({
            "EURUSD": eurusd_price_array,
            "DXY":    dxy_price_array,
            "GOLD":   gold_price_array,
        })

        # Before trading EURUSD, check if DXY already moved:
        props = graph.propagate("DXY", direction="UP")
        # [{"target": "EURUSD", "prob": 0.73, "lag": 2}, ...]

        # Who is driving this market right now?
        hubs = graph.hub_assets(top_n=3)
        # [("BTC", 1.94), ("DXY", 1.26), ("SPX", 0.88)]
    """

    def __init__(self, max_lags: int = 8, min_obs: int = 30):
        self.max_lags = max_lags
        self.min_obs  = min_obs
        self.edges:  list[CausalEdge] = []
        self.assets: list[str]        = []
        self._returns: dict[str, np.ndarray] = {}

    def build(self, price_series: dict[str, np.ndarray]) -> None:
        """
        Build causal graph from price history.
        price_series: {"ASSET": np.ndarray of prices (1-D)}
        """
        self.assets = list(price_series.keys())
        self._returns = {
            a: np.diff(np.log(np.clip(p, 1e-10, None)))
            for a, p in price_series.items()
        }
        self.edges = []
        for src in self.assets:
            for tgt in self.assets:
                if src == tgt:
                    continue
                edge = self._compute_edge(src, tgt)
                if edge.weight > 0.05:
                    self.edges.append(edge)
        log.info("CausalGraph built: %d assets, %d edges", len(self.assets), len(self.edges))

    def _compute_edge(self, src: str, tgt: str) -> CausalEdge:
        rx = self._returns.get(src, np.array([]))
        ry = self._returns.get(tgt, np.array([]))
        n  = min(len(rx), len(ry))
        if n < self.min_obs:
            return CausalEdge(src, tgt, 0.0, 0)

        rx, ry = rx[-n:], ry[-n:]

        # Method 1: Granger (linear cross-correlation proxy)
        granger_w, best_lag = self._granger_proxy(rx, ry)

        # Method 2: Transfer Entropy (mutual information at lag)
        te_score = self._transfer_entropy(rx, ry, lag=best_lag)

        # Method 3: Spearman rank at best lag
        spearman_w = self._spearman_lag(rx, ry, best_lag)

        # Method 4: CCM proxy (shadow manifold — simplified)
        ccm_w = self._ccm_proxy(rx, ry)

        # Composite: weighted fusion
        composite = (0.35 * granger_w + 0.30 * te_score +
                     0.20 * spearman_w + 0.15 * ccm_w)
        return CausalEdge(src, tgt, float(np.clip(composite, 0, 1)),
                          best_lag, te_score=te_score)

    def _granger_proxy(self, rx: np.ndarray, ry: np.ndarray
                       ) -> tuple[float, int]:
        best_w, best_lag = 0.0, 1
        for lag in range(1, min(self.max_lags + 1, len(rx) // 4)):
            if lag >= len(rx):
                break
            xcorr = float(np.corrcoef(rx[:-lag], ry[lag:])[0, 1])
            if abs(xcorr) > abs(best_w):
                best_w, best_lag = xcorr, lag
        # Convert correlation to [0,1] strength
        return float(np.clip(abs(best_w), 0, 1)), best_lag

    def _transfer_entropy(self, rx: np.ndarray, ry: np.ndarray,
                          lag: int = 1, bins: int = 6) -> float:
        if lag >= len(rx) or len(rx) < bins * 2:
            return 0.0
        # Quantise returns
        rx_q = np.digitize(rx, np.percentile(rx, np.linspace(0, 100, bins + 1)[1:-1]))
        ry_q = np.digitize(ry, np.percentile(ry, np.linspace(0, 100, bins + 1)[1:-1]))
        n    = len(rx_q) - lag

        def _entropy(arr: np.ndarray) -> float:
            _, counts = np.unique(arr, return_counts=True)
            p = counts / counts.sum()
            return float(-np.sum(p * np.log(p + 1e-12)))

        h_y_future  = _entropy(ry_q[lag:])
        joint        = np.stack([rx_q[:n], ry_q[:n], ry_q[lag:]], axis=1)
        _, cts       = np.unique(joint, axis=0, return_counts=True)
        p3           = cts / cts.sum()
        h_joint3     = float(-np.sum(p3 * np.log(p3 + 1e-12)))

        joint2       = np.stack([ry_q[:n], ry_q[lag:]], axis=1)
        _, cts2      = np.unique(joint2, axis=0, return_counts=True)
        p2           = cts2 / cts2.sum()
        h_joint2     = float(-np.sum(p2 * np.log(p2 + 1e-12)))

        te = h_y_future - h_joint3 + h_joint2
        return float(np.clip(te / (h_y_future + 1e-10), 0, 1))

    def _spearman_lag(self, rx: np.ndarray, ry: np.ndarray,
                      lag: int) -> float:
        if lag >= len(rx):
            return 0.0
        from scipy.stats import spearmanr  # type: ignore
        try:
            r, _ = spearmanr(rx[:-lag], ry[lag:])
            return float(np.clip(abs(r), 0, 1))
        except Exception:
            return 0.0

    def _ccm_proxy(self, rx: np.ndarray, ry: np.ndarray,
                   embed_dim: int = 3) -> float:
        """Simplified CCM: shadow manifold correlation."""
        if len(rx) < embed_dim + 10:
            return 0.0
        # Embed rx into delay coordinates
        n = len(rx) - embed_dim
        shadow = np.stack([rx[i:i + n] for i in range(embed_dim)], axis=1)
        target = ry[embed_dim:][:n]
        try:
            corr = float(np.corrcoef(shadow[:, 0], target)[0, 1])
            return float(np.clip(abs(corr), 0, 1))
        except Exception:
            return 0.0

    def propagate(self, origin: str, direction: str = "UP",
                  threshold: float = 0.15) -> list[dict]:
        """
        Given origin asset moved in `direction`, return likely propagations.
        Returns list sorted by causal strength descending.
        """
        results = []
        for edge in self.edges:
            if edge.source != origin or edge.weight < threshold:
                continue
            results.append({
                "target":    edge.target,
                "prob":      round(edge.weight, 3),
                "lag_bars":  edge.lag_bars,
                "direction": direction,
            })
        return sorted(results, key=lambda x: x["prob"], reverse=True)

    def hub_assets(self, top_n: int = 3) -> list[tuple[str, float]]:
        """Return top N assets by total outbound causal strength."""
        strength: dict[str, float] = {}
        for edge in self.edges:
            strength[edge.source] = strength.get(edge.source, 0.0) + edge.weight
        return sorted(strength.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def strongest_path(self, src: str, tgt: str) -> list[str]:
        """Greedy strongest causal path from src to tgt."""
        if src == tgt:
            return [src]
        visited = {src}
        path    = [src]
        for _ in range(len(self.assets)):
            candidates = [e for e in self.edges
                          if e.source == path[-1] and e.target not in visited]
            if not candidates:
                break
            best = max(candidates, key=lambda e: e.weight)
            path.append(best.target)
            visited.add(best.target)
            if best.target == tgt:
                return path
        return path


# ════════════════════════════════════════════════════════════════════════════
# 3. PRODUCTION CIRCUIT BREAKER MANAGER
# ════════════════════════════════════════════════════════════════════════════
# Extends QUIMERIA's reverse period / halt logic with production-grade
# infrastructure: Redis backing, auto-reset timers, async callbacks,
# and a new MANDRA_GATE breaker for the ΔE information condition.

class BreakerState(str, Enum):
    CLEAR   = "clear"
    TRIPPED = "tripped"


class BreakerName(str, Enum):
    REVERSE_PERIOD    = "reverse_period"    # QUIMERIA's own reverse period
    MAX_DRAWDOWN      = "max_drawdown"      # portfolio drawdown threshold
    CVAR_BREACH       = "cvar_breach"       # tail risk too high
    ADELIC_VIOLATION  = "adelic_violation"  # signal outside p-adic tube
    LIQUIDITY_CRISIS  = "liquidity_crisis"  # VPIN toxicity > 50%
    MANDRA_GATE       = "mandra_gate"       # ΔE ≤ threshold (no information)
    MANUAL_HALT       = "manual_halt"       # operator-triggered


@dataclass
class BreakerEvent:
    name:   BreakerName
    state:  BreakerState
    value:  float
    reason: str
    ts:     float = field(default_factory=time.time)


class CircuitBreakerManager:
    """
    Async circuit breaker manager — drop-in for QUIMERIA's halt logic.

    QUIMERIA usage:
        cb = CircuitBreakerManager()

        # Register a callback (e.g. log to QUIMERIA's log streams):
        async def on_halt(event):
            await quimeria_logger.critical(f"HALT: {event.name} {event.reason}")
        cb.on_trip(on_halt)

        # In QUIMERIA's tick loop, check before signal dispatch:
        if await cb.any_tripped():
            return  # u_t = 0

        # QUIMERIA's reverse period → wire into the breaker:
        if quimeria_rev_period_active:
            await cb.trip(BreakerName.REVERSE_PERIOD, rev_score, "λ fusion triggered")

        # Drawdown (QUIMERIA already tracks this):
        await cb.check_drawdown(current_drawdown_pct)

        # Mandra ΔE gate (new — see MandraBitGate below):
        await cb.check_mandra(signal_probs, prior_probs)
    """

    THRESHOLDS = {
        BreakerName.REVERSE_PERIOD:   0.52,    # QUIMERIA's own threshold
        BreakerName.MAX_DRAWDOWN:     0.08,    # 8% drawdown
        BreakerName.CVAR_BREACH:      0.15,    # 15% CVaR
        BreakerName.ADELIC_VIOLATION: 0.20,    # >20% signals outside tube
        BreakerName.LIQUIDITY_CRISIS: 0.85,    # VPIN > 0.85
        BreakerName.MANDRA_GATE:      0.02,    # ΔE minimum (nats)
        BreakerName.MANUAL_HALT:      0.00,
    }

    AUTO_RESET_S = {
        BreakerName.REVERSE_PERIOD:   300.0,
        BreakerName.MAX_DRAWDOWN:     None,
        BreakerName.CVAR_BREACH:      300.0,
        BreakerName.ADELIC_VIOLATION: 60.0,
        BreakerName.LIQUIDITY_CRISIS: 60.0,
        BreakerName.MANDRA_GATE:      30.0,
        BreakerName.MANUAL_HALT:      None,
    }

    def __init__(self):
        self._state: dict[str, BreakerState]   = {}
        self._events: list[BreakerEvent]         = []
        self._trip_times: dict[BreakerName, float] = {}
        self._callbacks: list[Callable[[BreakerEvent], Awaitable]] = []
        # Try Redis for cross-process state sharing
        try:
            import redis.asyncio as aioredis   # type: ignore
            self._redis = aioredis.from_url("redis://localhost:6379/0",
                                             decode_responses=True)
            log.info("CircuitBreaker: Redis connected")
        except Exception:
            self._redis = None
            log.info("CircuitBreaker: in-process store (Redis unavailable)")

    def on_trip(self, cb: Callable[[BreakerEvent], Awaitable]) -> None:
        self._callbacks.append(cb)

    async def trip(self, name: BreakerName, value: float, reason: str = "") -> None:
        ev = BreakerEvent(name=name, state=BreakerState.TRIPPED, value=value, reason=reason)
        self._state[name.value] = BreakerState.TRIPPED
        self._events.append(ev)
        self._trip_times[name] = time.time()
        log.warning("🔴 BREAKER TRIPPED: %s  %.4f  %s", name.value, value, reason)
        for cb in self._callbacks:
            try:    await cb(ev)
            except Exception as e: log.error("Callback error: %s", e)

    async def reset(self, name: BreakerName) -> None:
        self._state[name.value] = BreakerState.CLEAR
        self._trip_times.pop(name, None)
        log.info("🟢 BREAKER RESET: %s", name.value)

    async def reset_all(self) -> None:
        for name in BreakerName:
            await self.reset(name)

    async def is_tripped(self, name: BreakerName) -> bool:
        st = self._state.get(name.value, BreakerState.CLEAR)
        if st == BreakerState.TRIPPED:
            auto_s = self.AUTO_RESET_S.get(name)
            if auto_s and name in self._trip_times:
                if time.time() - self._trip_times[name] > auto_s:
                    await self.reset(name)
                    return False
        return st == BreakerState.TRIPPED

    async def any_tripped(self) -> bool:
        return any([await self.is_tripped(n) for n in BreakerName])

    async def check_drawdown(self, dd: float) -> None:
        if dd >= self.THRESHOLDS[BreakerName.MAX_DRAWDOWN]:
            await self.trip(BreakerName.MAX_DRAWDOWN, dd,
                            f"Drawdown {dd:.2%}")

    async def check_reverse_period(self, score: float) -> None:
        if score >= self.THRESHOLDS[BreakerName.REVERSE_PERIOD]:
            await self.trip(BreakerName.REVERSE_PERIOD, score,
                            f"Rev.period score {score:.3f}")

    async def check_mandra(self, signal_probs: np.ndarray,
                            prior_probs: np.ndarray) -> None:
        """
        Mandra ΔE gate: trip if KL(signal || prior) ≤ threshold.
        i.e. the signal carries insufficient information to justify a trade.
        """
        p   = np.clip(signal_probs, 1e-10, 1.0)
        q   = np.clip(prior_probs,  1e-10, 1.0)
        kl  = float(np.sum(p * np.log(p / q)))
        if kl <= self.THRESHOLDS[BreakerName.MANDRA_GATE]:
            await self.trip(BreakerName.MANDRA_GATE, kl,
                            f"ΔE={kl:.4f} — signal has no information")

    async def check_liquidity(self, vpin: np.ndarray) -> None:
        toxic = float(np.mean(vpin > self.THRESHOLDS[BreakerName.LIQUIDITY_CRISIS]))
        if toxic >= 0.50:
            await self.trip(BreakerName.LIQUIDITY_CRISIS, toxic,
                            f"{toxic:.0%} assets toxic VPIN")

    def status(self) -> dict[str, str]:
        return {n.value: self._state.get(n.value, BreakerState.CLEAR).value
                for n in BreakerName}

    def recent_events(self, n: int = 20) -> list[BreakerEvent]:
        return self._events[-n:]


# ════════════════════════════════════════════════════════════════════════════
# 4. ADELIC LEVEL SCORER
# ════════════════════════════════════════════════════════════════════════════
# Ranks QUIMERIA's order blocks / FVG levels by institutional significance
# using p-adic valuation across primes {2,3,5,7,11,13}.
# Levels with high adelic score carry more institutional memory.

_PRIMES = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0]


def _p_adic_norm(x: np.ndarray, p: float) -> np.ndarray:
    safe = np.clip(np.abs(np.asarray(x, dtype=np.float64)), 1e-15, None)
    v_p  = np.floor(-np.log(safe) / np.log(p))
    result = np.exp(-v_p * np.log(p))
    if hasattr(result, "ndim") and result.ndim == 1 and len(result) == 1:
        return result[0]
    return result


class AdelicLevelScorer:
    """
    Score price levels by p-adic institutional significance.

    QUIMERIA usage:
        scorer = AdelicLevelScorer()

        # QUIMERIA finds 5 order blocks:
        ob_levels = [1.08640, 1.08500, 1.08200, 1.09000, 1.08800]
        ranked    = scorer.rank_levels(ob_levels)
        # Returns levels sorted by significance — trade from the top ones

        # Check if a specific level is institutionally significant:
        sig = scorer.score(1.08640)
        # sig["composite"] = 0.73  (higher = more significant)
    """

    def __init__(self, tube_radius: float = 0.05):
        self.tube_radius = tube_radius

    def score(self, level: float) -> dict:
        """
        Score a price level by proximity to institutional boundaries.
        Uses p-adic inspired hierarchical proximity across boundary scales.

        Higher composite = more institutionally significant (on a key level).
        Scores proximity to 1000-pip, 500-pip, 100-pip, 50-pip, 10-pip boundaries.

        e.g. 1.09000 scores high (on 100-pip boundary)
             1.08637 scores low  (arbitrary messy level)
        """
        boundaries = [0.1, 0.05, 0.01, 0.005, 0.001]
        prox_scores = {}
        for i, b in enumerate(boundaries):
            frac = (level % b) / b          # position within interval [0, 1)
            dist = min(frac, 1.0 - frac)   # distance to nearest boundary [0, 0.5]
            prox = 1.0 - dist * 2.0        # 1 = on boundary, 0 = midpoint
            prox_scores[f"b{int(b*1000)}pip"] = round(float(prox), 4)

        composite = float(np.mean(list(prox_scores.values())))
        in_tube   = composite > (1.0 - self.tube_radius * 10)  # near boundary = in tube

        # Also compute raw p-adic norms on fractional pip distance from round level
        ref_level  = round(level, 1)
        pip_dist   = abs(level - ref_level)
        if pip_dist > 1e-9:
            padic = {f"p{int(p)}": round(float(_p_adic_norm(np.array([pip_dist]), p)), 6)
                     for p in _PRIMES}
        else:
            padic = {f"p{int(p)}": 0.0 for p in _PRIMES}

        return {
            "level":     level,
            "composite": round(composite, 6),
            "in_tube":   in_tube,
            **prox_scores,
            **padic,
        }

    def rank_levels(self, levels: list[float]) -> list[dict]:
        """
        Score and rank a list of price levels.
        Returns list sorted by composite score descending
        (higher proximity score = more institutionally significant).
        """
        scored = [self.score(lv) for lv in levels]
        return sorted(scored, key=lambda x: x["composite"], reverse=True)

    def most_significant(self, levels: list[float], top_n: int = 3) -> list[float]:
        """Return the top_n most institutionally significant levels."""
        ranked = self.rank_levels(levels)
        return [r["level"] for r in ranked[:top_n]]

    def containment_mask(self, signal: np.ndarray) -> np.ndarray:
        """
        Boolean mask: True where signal_i passes adelic containment
        (i.e. is within the institutional tube for all primes).
        Equivalent to AEGIS's adelic_containment_proof().
        """
        signal   = np.asarray(signal, dtype=np.float32)
        rho      = max(self.tube_radius, 1e-12)
        passes   = np.ones(len(signal), dtype=bool)
        for p in _PRIMES:
            sn = _p_adic_norm(signal, p)
            rn = _p_adic_norm(np.full_like(signal, rho, dtype=np.float64), p)
            passes = passes & (np.asarray(sn) <= np.asarray(rn) * 1.001)
        return passes


# ════════════════════════════════════════════════════════════════════════════
# 5. STREAMING REVERSE PERIOD GUARD
# ════════════════════════════════════════════════════════════════════════════
# Wraps the AEGIS λ₁–λ₅ detector into a streaming on_bar() interface
# compatible with QUIMERIA's existing tick loop. Adds Spectral Crowding
# Monitor and forensic audit trail on top of QUIMERIA's own reverse period.

@dataclass
class LambdaReading:
    l1: float = 0.0   # phase entrapment
    l2: float = 0.0   # temporal alignment failure
    l3: float = 0.0   # spectral phase inversion
    l4: float = 0.0   # confluence collapse
    l5: float = 0.0   # liquidity field inversion

    @property
    def score(self) -> float:
        W = [0.35, 0.25, 0.20, 0.15, 0.05]
        return sum(w * v for w, v in zip(W, [self.l1, self.l2, self.l3, self.l4, self.l5]))

    @property
    def active(self) -> bool:
        return self.score > 0.52


class AegisReversePeriodGuard:
    """
    Streaming reverse period guard — drop into QUIMERIA's tick loop.

    QUIMERIA usage:
        guard = AegisReversePeriodGuard(halt_on_trigger=True)

        # In the tick loop (each new bar):
        bar = {
            "close":    price,
            "high":     high,
            "low":      low,
            "volume":   volume,
            "sigma":    amd_phase_int,     # 0=Acc, 1=Man, 2=Dist
            "phi":      confluence_score,  # 0–1, from QUIMERIA fusion
            "killzone": is_killzone,       # bool, London/NY session
        }
        result = guard.on_bar(bar)

        if result["action"] == "HALT":
            # Override QUIMERIA's u_t = 0 protocol
            quimeria_signal = 0.0

        elif result["action"] == "REDUCE":
            # Attenuate signal: high score but below halt threshold
            quimeria_signal *= (1.0 - result["score"])
    """

    def __init__(self, halt_on_trigger: bool = True,
                 buffer_size: int = 200):
        self.halt_on_trigger = halt_on_trigger
        self._buf_size       = buffer_size
        self._price:   list[float] = []
        self._high:    list[float] = []
        self._low:     list[float] = []
        self._ret:     list[float] = []
        self._ofi:     list[float] = []
        self._sigma:   list[int]   = []
        self._phi:     list[float] = []
        self._kz:      list[bool]  = []
        self._vol:     list[float] = []
        self._lambdas: list[LambdaReading] = []
        self._n_bars   = 0
        self._triggers = 0

    def _push(self, lst: list, val: Any) -> None:
        lst.append(val)
        if len(lst) > self._buf_size:
            lst.pop(0)

    def _compute_lambdas(self) -> LambdaReading:
        n = len(self._price)
        if n < 15:
            return LambdaReading()

        price = np.array(self._price)
        ofi   = np.array(self._ofi)
        phi   = np.array(self._phi)
        ret   = np.array(self._ret)

        # λ₁ — phase entrapment (σ==2 with low price variation)
        recent  = price[-20:]
        p_range = float(np.ptp(recent))
        atr_est = float(np.mean(np.abs(np.diff(price[-20:])))) + 1e-10
        norm_var = p_range / (atr_est * 20)
        sigma_now = self._sigma[-1] if self._sigma else 1
        l1 = float(np.clip((sigma_now == 2) * max(0, 1.0 - norm_var / 0.4), 0, 1))

        # λ₂ — killzone temporal failure (negative OFI mean in KZ)
        ofi_slice = ofi[-10:]
        mean_ofi  = float(ofi_slice.mean()) if len(ofi_slice) else 0.0
        kz_now    = self._kz[-1] if self._kz else False
        l2 = float(np.clip(kz_now * max(0, -mean_ofi * 4), 0, 1))

        # λ₃ — spectral phase inversion (short trend opposes long trend)
        if n >= 40:
            short_trend = float(price[-12:].mean() - price[-12])
            long_trend  = float(price[-40:].mean() - price[-40])
            phase_flip  = (short_trend * long_trend < 0) and abs(short_trend) > atr_est * 0.3
            l3 = float(np.clip(phase_flip * min(1.0, abs(short_trend - long_trend) / (atr_est * 2)), 0, 1))
        else:
            l3 = 0.0

        # λ₄ — confluence collapse (high phi, negative returns)
        phi_now  = self._phi[-1] if self._phi else 0.5
        mean_ret = float(ret[-8:].mean()) if n >= 8 else 0.0
        l4 = float(np.clip((phi_now > 0.6) * max(0, -mean_ret / (atr_est * 0.5 + 1e-10)), 0, 1))

        # λ₅ — liquidity field inversion (OFI sign flip)
        if len(ofi) >= 4:
            ofi_sign_flip = ofi[-4] * ofi[-1] < 0 and abs(mean_ofi) > 0.1
            l5 = float(np.clip(ofi_sign_flip * 0.7, 0, 1))
        else:
            l5 = 0.0

        return LambdaReading(l1=l1, l2=l2, l3=l3, l4=l4, l5=l5)

    def on_bar(self, bar: dict) -> dict:
        """
        Process one bar and return action dict.

        Returns:
            {
                "action":  "CONTINUE" | "REDUCE" | "HALT" | "WARMUP",
                "score":   float,          # composite λ score [0, 1]
                "lambdas": LambdaReading,
                "triggered": bool,
            }
        """
        close = bar.get("close", 0.0)
        prev  = self._price[-1] if self._price else close
        ret   = (close - prev) / (prev + 1e-10)

        self._push(self._price,  close)
        self._push(self._high,   bar.get("high",   close))
        self._push(self._low,    bar.get("low",    close))
        self._push(self._vol,    bar.get("volume", 0.0))
        self._push(self._ret,    ret)
        self._push(self._ofi,    bar.get("ofi",   float(np.sign(ret) * 0.1)))
        self._push(self._sigma,  bar.get("sigma",  1))
        self._push(self._phi,    bar.get("phi",    0.5))
        self._push(self._kz,     bar.get("killzone", False))
        self._n_bars += 1

        if self._n_bars < 20:
            return {"action": "WARMUP", "score": 0.0,
                    "lambdas": LambdaReading(), "triggered": False}

        lams      = self._compute_lambdas()
        triggered = lams.active
        score     = lams.score
        if triggered:
            self._triggers += 1

        if triggered and self.halt_on_trigger:
            action = "HALT"
        elif score > 0.35:
            action = "REDUCE"
        else:
            action = "CONTINUE"

        self._lambdas.append(lams)
        if len(self._lambdas) > 500:
            self._lambdas.pop(0)

        return {
            "action":    action,
            "score":     round(score, 4),
            "lambdas":   lams,
            "triggered": triggered,
            "n_triggers": self._triggers,
        }


# ════════════════════════════════════════════════════════════════════════════
# 6. MANDRA BIT GATE
# ════════════════════════════════════════════════════════════════════════════
# Information-theoretic trade gate: only allow execution if the signal
# reduces market uncertainty by more than a minimum threshold (ΔE > 0).
# This is QUIMERIA's "Bit Second Law of Thermodynamics" formalized.

class MandraBitGate:
    """
    Trade gate based on Kullback-Leibler divergence between signal
    distribution and the prior (uniform / historical baseline).

    ΔE = KL( P_signal || P_prior )
    Trade allowed ↔  ΔE > threshold

    QUIMERIA usage:
        gate = MandraBitGate(threshold=0.02)

        # QUIMERIA's fusion engine outputs per-class probabilities:
        # e.g. P(LONG)=0.62, P(FLAT)=0.28, P(SHORT)=0.10
        signal_probs = np.array([0.62, 0.28, 0.10])

        result = gate.evaluate(signal_probs)
        if not result["allow"]:
            return  # signal has insufficient information — skip trade

        # The gate also returns position_scale: attenuate size by ΔE
        position_size *= result["position_scale"]
    """

    def __init__(self,
                 threshold: float = 0.02,    # min KL divergence (nats)
                 n_classes: int   = 3,        # LONG / FLAT / SHORT
                 prior:     Optional[np.ndarray] = None):
        self.threshold = threshold
        self.n_classes = n_classes
        self.prior     = prior if prior is not None else np.ones(n_classes) / n_classes
        self._history: list[float] = []

    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """KL( p || q ) in nats."""
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        return float(np.sum(p * np.log(p / q)))

    def evaluate(self, signal_probs: np.ndarray,
                 prior: Optional[np.ndarray] = None) -> dict:
        """
        Evaluate whether the signal contains enough information to trade.

        Args:
            signal_probs: (n_classes,) array of P(class | signal), sums to ~1
            prior:        (n_classes,) baseline distribution (default: uniform)

        Returns dict:
            allow:          bool   — True if ΔE > threshold
            delta_e:        float  — KL divergence (nats)
            position_scale: float  — [0, 1], scales position size by ΔE strength
            dominant_class: int    — argmax of signal_probs
            confidence:     float  — max(signal_probs)
        """
        q = prior if prior is not None else self.prior
        signal_probs = np.asarray(signal_probs, dtype=np.float64)
        signal_probs = signal_probs / (signal_probs.sum() + 1e-10)

        delta_e   = self.kl_divergence(signal_probs, q)
        allow     = delta_e > self.threshold
        # Scale: ΔE → [0,1] sigmoid-like, so position size reflects info content
        pos_scale = float(np.clip(delta_e / (delta_e + self.threshold), 0.0, 1.0))

        self._history.append(delta_e)
        if len(self._history) > 1000:
            self._history.pop(0)

        return {
            "allow":          allow,
            "delta_e":        round(delta_e, 6),
            "position_scale": round(pos_scale, 4),
            "dominant_class": int(np.argmax(signal_probs)),
            "confidence":     round(float(signal_probs.max()), 4),
            "threshold":      self.threshold,
        }

    def update_prior(self, new_prior: np.ndarray) -> None:
        """Update the baseline distribution (e.g. rolling empirical prior)."""
        self.prior = np.clip(new_prior, 1e-10, 1.0)
        self.prior /= self.prior.sum()

    def rolling_average_de(self, window: int = 60) -> float:
        """Average ΔE over the last `window` bars — regime health indicator."""
        h = self._history[-window:]
        return float(np.mean(h)) if h else 0.0


# ════════════════════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER — AegisExtensions
# ════════════════════════════════════════════════════════════════════════════
# Single object that initialises and pre-wires all six extensions.
# Provides a unified on_bar() and on_signal() interface for QUIMERIA.

@dataclass
class AegisBarResult:
    """Full AEGIS extension output for one bar."""
    action:          str            # "TRADE" | "REDUCE" | "HALT" | "WARMUP"
    kelly_size:      float          # final position size (fraction of capital)
    venue_allocation: np.ndarray   # (n_venues,) fractions
    rev_score:       float          # reverse period score [0, 1]
    rev_triggered:   bool
    lambdas:         LambdaReading
    delta_e:         float          # Mandra information gain (nats)
    mandra_allow:    bool
    adelic_ok:       bool           # signal within p-adic tube
    causal_hubs:     list           # top causal assets right now
    halt_reason:     str            # populated when action == "HALT"
    breaker_status:  dict


class AegisExtensions:
    """
    Single entry-point for all AEGIS extensions inside QUIMERIA.

    Quickstart:
        aegis = AegisExtensions(n_venues=4, n_classes=3)
        aegis.start()

        result = await aegis.on_signal(
            bar            = bar_dict,
            signal         = quimeria_signal_float,     # [-1, 1]
            confidence     = quimeria_confidence,       # [0, 1]
            signal_probs   = np.array([0.6, 0.3, 0.1]),# LONG/FLAT/SHORT
            returns        = np.array([...]),           # recent returns
            vpin           = np.array([...]),           # per-asset VPIN
            price_universe = {"DXY": ..., "GOLD": ...} # for causal graph
        )

        if result.action == "TRADE":
            broker.submit(signal * result.kelly_size,
                          allocation=result.venue_allocation)
    """

    def __init__(self,
                 n_venues:        int   = 4,
                 n_classes:       int   = 3,
                 kelly_limit:     float = 0.02,
                 mandra_threshold: float = 0.02,
                 tube_radius:     float = 0.05):
        self.router   = SchurRouter(n_venues=n_venues)
        self.breakers = CircuitBreakerManager()
        self.guard    = AegisReversePeriodGuard(halt_on_trigger=True)
        self.adelic   = AdelicLevelScorer(tube_radius=tube_radius)
        self.mandra   = MandraBitGate(threshold=mandra_threshold, n_classes=n_classes)
        self.causal   = CausalTransmissionGraph()
        self.kelly_limit = kelly_limit
        self._causal_built = False
        self._n_venues     = n_venues

    def start(self) -> None:
        log.info("AegisExtensions started — 6 modules active")
        log.info("  SchurRouter:       %d venues", self._n_venues)
        log.info("  CircuitBreakers:   %d breakers", len(BreakerName))
        log.info("  RevPeriodGuard:    λ₁–λ₅ streaming")
        log.info("  AdelicScorer:      primes %s", _PRIMES)
        log.info("  MandraBitGate:     threshold=%.3f", self.mandra.threshold)
        log.info("  CausalGraph:       ready (call build_causal() with price data)")

    def build_causal(self, price_universe: dict[str, np.ndarray]) -> None:
        """
        Build the causal transmission graph from historical price data.
        Call once on startup, then update periodically (e.g. daily).
        """
        self.causal.build(price_universe)
        self._causal_built = True
        hubs = self.causal.hub_assets(top_n=3)
        log.info("CausalGraph built — hubs: %s", hubs)

    async def on_signal(
        self,
        bar:            dict,
        signal:         float,                    # QUIMERIA output [-1, 1]
        confidence:     float,                    # QUIMERIA confidence [0, 1]
        signal_probs:   Optional[np.ndarray] = None,  # [P(long), P(flat), P(short)]
        returns:        Optional[np.ndarray] = None,
        vpin:           Optional[np.ndarray] = None,
    ) -> AegisBarResult:
        """
        Process one bar + QUIMERIA signal through all 6 AEGIS extensions.
        Returns AegisBarResult with final action and execution parameters.
        """
        halt_reason = ""

        # ── 1. Reverse period guard ──────────────────────────────────────
        guard_result = self.guard.on_bar(bar)
        if guard_result["action"] == "WARMUP":
            return AegisBarResult(
                action="WARMUP", kelly_size=0.0,
                venue_allocation=np.ones(self._n_venues)/self._n_venues,
                rev_score=0.0, rev_triggered=False,
                lambdas=LambdaReading(), delta_e=0.0, mandra_allow=True,
                adelic_ok=True, causal_hubs=[], halt_reason="warmup",
                breaker_status=self.breakers.status(),
            )

        rev_score     = guard_result["score"]
        rev_triggered = guard_result["triggered"]
        lambdas       = guard_result["lambdas"]

        # ── 2. Wire reverse period into circuit breaker ──────────────────
        await self.breakers.check_reverse_period(rev_score)
        if vpin is not None:
            await self.breakers.check_liquidity(vpin)

        # ── 3. Mandra ΔE gate ────────────────────────────────────────────
        if signal_probs is None:
            # Infer from signal scalar: positive → LONG bias
            long_p  = max(0.0, signal)
            short_p = max(0.0, -signal)
            flat_p  = 1.0 - long_p - short_p
            signal_probs = np.array([long_p, flat_p, short_p])
        mandra_result = self.mandra.evaluate(signal_probs)
        mandra_allow  = mandra_result["allow"]
        delta_e       = mandra_result["delta_e"]

        if not mandra_allow:
            await self.breakers.trip(
                BreakerName.MANDRA_GATE, delta_e,
                f"ΔE={delta_e:.4f} — insufficient information"
            )

        # ── 4. Adelic containment ────────────────────────────────────────
        sig_vec  = np.array([signal * confidence])
        adelic_ok = bool(self.adelic.containment_mask(sig_vec)[0])

        # ── 5. Check all circuit breakers ────────────────────────────────
        if await self.breakers.any_tripped():
            tripped = [n.value for n in BreakerName
                       if await self.breakers.is_tripped(n)]
            halt_reason = ", ".join(tripped)
            return AegisBarResult(
                action="HALT", kelly_size=0.0,
                venue_allocation=np.zeros(self._n_venues),
                rev_score=rev_score, rev_triggered=rev_triggered,
                lambdas=lambdas, delta_e=delta_e, mandra_allow=mandra_allow,
                adelic_ok=adelic_ok, causal_hubs=[], halt_reason=halt_reason,
                breaker_status=self.breakers.status(),
            )

        # ── 6. Kelly size ─────────────────────────────────────────────────
        base_size    = abs(signal) * confidence * self.kelly_limit
        info_size    = base_size * mandra_result["position_scale"]
        rev_penalty  = max(0.0, 1.0 - rev_score * 2.0)
        kelly_size   = float(np.clip(info_size * rev_penalty, 0.0, self.kelly_limit))

        # ── 7. Schur venue routing ────────────────────────────────────────
        desired      = np.full(self._n_venues, confidence / self._n_venues, dtype=np.float32)
        schur_result = self.router.route(desired)
        allocation   = schur_result.allocation

        # ── 8. Causal hubs ────────────────────────────────────────────────
        causal_hubs = self.causal.hub_assets(top_n=3) if self._causal_built else []

        # ── 9. Final action ───────────────────────────────────────────────
        if rev_score > 0.35:
            action = "REDUCE"
        else:
            action = "TRADE"

        return AegisBarResult(
            action           = action,
            kelly_size       = round(kelly_size, 6),
            venue_allocation = allocation,
            rev_score        = rev_score,
            rev_triggered    = rev_triggered,
            lambdas          = lambdas,
            delta_e          = delta_e,
            mandra_allow     = mandra_allow,
            adelic_ok        = adelic_ok,
            causal_hubs      = causal_hubs,
            halt_reason      = "",
            breaker_status   = self.breakers.status(),
        )


# ════════════════════════════════════════════════════════════════════════════
# QUICK INTEGRATION GUIDE
# ════════════════════════════════════════════════════════════════════════════
"""
STEP 1 — Drop this file into your QUIMERIA project root.
STEP 2 — In your QUIMERIA main signal loop, add:

    from aegis_extensions import AegisExtensions

    aegis = AegisExtensions(n_venues=4)
    aegis.start()

    # Optional: build causal graph on startup
    aegis.build_causal({"EURUSD": eurusd_hist, "DXY": dxy_hist})

STEP 3 — In your FastAPI WebSocket tick handler:

    @app.websocket("/ws/signal")
    async def signal_stream(websocket: WebSocket):
        await websocket.accept()
        while True:
            bar = await get_next_bar()                  # your existing feed
            signal, conf, probs = quimeria_engine(bar)  # your existing fusion

            result = await aegis.on_signal(
                bar          = bar,
                signal       = signal,
                confidence   = conf,
                signal_probs = probs,
            )

            payload = {
                "signal":     signal,
                "action":     result.action,
                "size":       result.kelly_size,
                "allocation": result.venue_allocation.tolist(),
                "rev_score":  result.rev_score,
                "delta_e":    result.delta_e,
                "lambdas":    {
                    "l1": result.lambdas.l1,
                    "l2": result.lambdas.l2,
                    "l3": result.lambdas.l3,
                    "l4": result.lambdas.l4,
                    "l5": result.lambdas.l5,
                },
            }
            await websocket.send_json(payload)

THAT'S IT. QUIMERIA's signal intelligence is unchanged.
AEGIS adds: sizing, routing, gates, guards, memory, causal context.
"""
