"""
logger.py  —  QUIMERIA SMK event logging.
Writes all bar events, veto decisions, and signals to rotating log files.
Logs folder: smk_ipda_trading_system/logs/
"""
import os
import json
import logging
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Optional

# ── LOG FOLDER ────────────────────────────────────────────────────────────────

def _find_log_dir() -> str:
    """Find or create the logs/ folder relative to the project root."""
    here = os.path.dirname(os.path.abspath(__file__))        # backend/
    root = os.path.dirname(here)                              # project root
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

LOG_DIR = _find_log_dir()


# ── LOGGERS ───────────────────────────────────────────────────────────────────

def _make_logger(name: str, filename: str, max_mb: int = 10, backups: int = 5) -> logging.Logger:
    path = os.path.join(LOG_DIR, filename)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = RotatingFileHandler(
            path,
            maxBytes=max_mb * 1024 * 1024,
            backupCount=backups,
            encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


# Three log streams — mirrors the three frontend panels
_event_log  = _make_logger("smk.events",  "events.log")     # FVGs, OBs, reversals
_veto_log   = _make_logger("smk.veto",    "veto.log")       # Ring 0 decisions
_trade_log  = _make_logger("smk.trades",  "trades.log")     # entries, exits, P&L
_session_log= _make_logger("smk.session", "session.log")    # startup, shutdown, data loads
_raw_log    = _make_logger("smk.raw",     "raw_bars.log",   max_mb=50)  # full bar JSON

# Trade ID counter — persists for session lifetime
_trade_counter = 0
_open_trades: dict = {}   # trade_id → {side, price, lots, symbol, sl, tp}


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def log_session(msg: str):
    """Log session events: startup, data loads, connections."""
    _session_log.info(f"[{_ts()}] {msg}")
    print(f"[LOG] {msg}")


def log_bar(bar_result: dict):
    """
    Called after every step(). Extracts and logs:
    - Veto decisions to veto.log
    - Notable events (FVGs, OBs, reversals, Judas swings) to events.log
    - Full JSON to raw_bars.log
    """
    if not bar_result:
        return

    idx   = bar_result.get("bar_index", 0)
    bar   = bar_result.get("bar", {})
    ts    = datetime.fromtimestamp(bar.get("time", 0), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    close = bar.get("close", 0)
    label = f"[{str(idx).zfill(4)}] {ts} C={close:.5f}"

    # ── VETO LOG ──────────────────────────────────────────────────────────────
    veto     = bar_result.get("veto", {})
    decision = veto.get("decision", "")
    reasons  = veto.get("reasons", [])
    fusion   = bar_result.get("fusion", {})
    amd      = bar_result.get("amd", {})

    _veto_log.info(
        f"[{_ts()}] {label} | "
        f"AMD={amd.get('state','?'):12s} | "
        f"DECISION={decision:8s} | "
        f"p={fusion.get('p_fused', 0):+.3f} | "
        f"conf={fusion.get('confidence', 0):.2f} | "
        f"reasons={','.join(reasons) if reasons else 'none'}"
    )

    # ── EVENT LOG — notable signals only ─────────────────────────────────────
    events = []

    # AMD phase change
    if amd.get("changed"):
        prev  = amd.get("prev", "")
        state = amd.get("state", "")
        events.append(f"AMD_TRANSITION {prev}->{state}")

    if amd.get("R_MASTER"):
        events.append("R_MASTER_RESET")

    # Manipulation / Judas swing
    manip = bar_result.get("manipulation", {})
    if manip.get("active"):
        events.append(f"JUDAS_SWING score={manip.get('score', 0):.0f} wick={manip.get('wick', 0):.5f}")

    # FVGs
    fvg = bar_result.get("fvg", {})
    for g in fvg.get("recent", []):
        events.append(f"FVG {g.get('type','?')} eq={g.get('eq', 0):.5f}")

    # Harmonic trap
    harmonic = bar_result.get("harmonic", {})
    if harmonic.get("inverted"):
        events.append(f"HARMONIC_TRAP phi={harmonic.get('phase_diff', 0):.3f}")

    # Strong fusion signal
    p_fused = fusion.get("p_fused", 0)
    conf    = fusion.get("confidence", 0)
    if abs(p_fused) > 0.45 and decision == "Proceed":
        direction = "LONG" if p_fused > 0 else "SHORT"
        events.append(f"SIGNAL_{direction} p={p_fused:+.3f} conf={conf:.2f}")

    # KL fracture
    kl = bar_result.get("kl", {})
    if not kl.get("stable") and kl.get("score", 0) > 0.8:
        events.append(f"KL_FRACTURE score={kl.get('score', 0):.3f}")

    # Topology fracture
    topo = bar_result.get("topology", {})
    if topo.get("fractured"):
        events.append(f"TOPO_FRACTURE h1={topo.get('h1_score', 0):.2f}")

    # Phase entrapment critical mass
    vd = bar_result.get("vol_decay", {})
    if vd.get("entrapped") and vd.get("stasis", 0) > 0 and vd.get("stasis", 0) % 5 == 0:
        events.append(f"ENTRAPMENT stasis={vd.get('stasis', 0)} energy={vd.get('energy', 0):.1f}")

    for ev in events:
        _event_log.info(f"[{_ts()}] {label} | {ev}")

    # ── TRADE LOG — fire on TRADE action from execution bridge ───────────────
    exe = bar_result.get("execution", {})
    if exe:
        _log_execution(exe, label, bar.get("close", 0))

    # ── RAW BAR LOG ───────────────────────────────────────────────────────────
    try:
        _raw_log.info(json.dumps(bar_result, separators=(',', ':')))
    except Exception:
        pass


def _log_execution(exe: dict, label: str, close: float):
    """
    Internal: called from log_bar() to write trade open/close entries.
    Tracks open positions in _open_trades so it can log P&L on close.
    """
    global _trade_counter, _open_trades

    action   = exe.get("action", "")
    is_armed = exe.get("is_armed", False)
    direction= exe.get("direction", 0)
    sl       = exe.get("stop_loss_price", 0.0)
    tp       = exe.get("take_profit_price", 0.0)
    lots     = exe.get("lot_size", 0.0)
    pips     = exe.get("risk_pips", 0.0)
    rr       = exe.get("rr_ratio", 0.0)
    pattern  = exe.get("pattern", "")
    dominant = exe.get("dominant", "X")
    kelly    = exe.get("kelly_size", 0.0)
    de       = exe.get("delta_e", 0.0)

    side = "LONG" if direction == 1 else "SHORT" if direction == -1 else "FLAT"

    # ── TRADE OPEN ────────────────────────────────────────────────────────────
    if action == "TRADE" and is_armed and lots > 0:
        _trade_counter += 1
        tid = _trade_counter
        _open_trades[tid] = {
            "side":  side,
            "price": close,
            "lots":  lots,
            "sl":    sl,
            "tp":    tp,
        }
        _trade_log.info(
            f"[{_ts()}] OPEN   T{tid:04d} {side:5s} {lots:.2f}L "
            f"@ {close:.5f} | "
            f"SL={sl:.5f} TP={tp:.5f} | "
            f"pips={pips:.1f} RR={rr:.2f} | "
            f"pattern=[{dominant}]{pattern} kelly={kelly:.4f} ΔE={de:+.4f} | "
            f"{label}"
        )

    # ── TRADE CLOSE / HALT — check if any open position just got stopped ──────
    # Close is inferred when a previously TRADE bar transitions to HALT/REDUCE
    # and we have an open position. Calculate P&L against current close price.
    elif action in ("HALT", "REDUCE") and _open_trades:
        for tid, pos in list(_open_trades.items()):
            entry   = pos["price"]
            pos_dir = 1 if pos["side"] == "LONG" else -1
            pnl_pips = (close - entry) * pos_dir / 0.0001
            reason = exe.get("reason", action)
            _trade_log.info(
                f"[{_ts()}] CLOSE  T{tid:04d} {pos['side']:5s} {pos['lots']:.2f}L "
                f"@ {close:.5f} | "
                f"PNL={pnl_pips:+.1f}pips | "
                f"reason={reason} | "
                f"{label}"
            )
            del _open_trades[tid]


def log_trade(action: str, trade_id: int, side: str, price: float,
              lots: float, pnl_pips: Optional[float] = None,
              symbol: str = "EURUSD"):
    """Log trade events: open, close."""
    if pnl_pips is not None:
        _trade_log.info(
            f"[{_ts()}] CLOSE  T{trade_id:04d} {side.upper():4s} {lots:.2f}lots "
            f"@ {price:.5f} | PNL={pnl_pips:+.1f}pips | {symbol}"
        )
    else:
        _trade_log.info(
            f"[{_ts()}] OPEN   T{trade_id:04d} {side.upper():4s} {lots:.2f}lots "
            f"@ {price:.5f} | {symbol}"
        )


def log_data_load(source: str, count: int, symbol: str = ""):
    """Log data source loads."""
    log_session(f"DATA LOAD | source={source} bars={count} symbol={symbol}")


def get_log_dir() -> str:
    return LOG_DIR


def get_log_files() -> list:
    """Return list of log files with sizes."""
    files = []
    for f in sorted(os.listdir(LOG_DIR)):
        if f.endswith(".log"):
            path = os.path.join(LOG_DIR, f)
            size = os.path.getsize(path)
            files.append({
                "name": f,
                "path": path,
                "size_kb": round(size / 1024, 1),
                "modified": datetime.fromtimestamp(
                    os.path.getmtime(path), tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M UTC")
            })
    return files
