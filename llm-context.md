# QUIMERIA / SOVEREIGN MARKET KERNEL — LLM CONTEXT DOCUMENT

> **Purpose:** This document gives any AI agent full working context to continue development on the QUIMERIA system without needing prior conversation history. Read this before touching any file.

---

## 1. WHAT THIS SYSTEM IS

**QUIMERIA** is a full-stack algorithmic trading analysis platform built around the **Sovereign Market Kernel (SMK)** — a proprietary multi-layer signal processing engine that models institutional price delivery as a deterministic, structured process. It is not a random-walk system. It treats markets as a planned delivery algorithm (IPDA framework).

**Core thesis:** Institutional markets follow Accumulation → Manipulation → Distribution → Retracement cycles. QUIMERIA decodes these in real time across 18+ parallel detector modules and fuses outputs into a single executable signal with Ring 0 veto authority.

**Status:** Working proof-of-concept demo. All 18 core SMK modules load on Roberto's machine with JAX available. Plugin layer (6 plugins) loads and runs. AEGIS execution bridge is wired but `stop_loss_manager.py` and `aegis_extensions.py` must be present for full execution chain.

---

## 2. REPOSITORY STRUCTURE

```
smk_ipda_trading_system/              ← PROJECT ROOT
├── start.bat                         ← Windows launcher (sets SMK_DIR, starts uvicorn)
├── start.sh                          ← Linux launcher
├── aegis_extensions.py               ← AEGIS integration layer (SchurRouter, MandraBitGate etc)
├── lambda_fusion_engine.py           ← Ring 0 fusion engine
├── SovereignMarketKernel.py          ← Top-level orchestrator (legacy, not used by FastAPI)
├── stop_loss_manager.py              ← WRONG LOCATION — should be in backend/
│
├── backend/                          ← FastAPI + SMK pipeline
│   ├── main.py                       ← FastAPI app, all REST + WebSocket endpoints
│   ├── smk_pipeline.py               ← 18-module bar-by-bar SMK detector pipeline
│   ├── data_connectors.py            ← CSV parsers (MT4/MT5/TV/Dukascopy), Bitget, OANDA, sample gen
│   ├── realtime.py                   ← Bitget REST polling live feed
│   ├── logger.py                     ← Rotating log files (events/veto/trades/session/raw)
│   ├── plugin_manager.py             ← Plugin discovery, loading, per-bar execution
│   ├── aegis_bridge.py               ← AEGIS execution bridge (CLM→SLM→AegisExtensions)
│   ├── jax_shim.py                   ← numpy shim when JAX not installed
│   ├── statsmodels_shim.py           ← stub VAR when statsmodels not installed
│   ├── fix_pipeline.py               ← Run once to write clean smk_pipeline.py to disk
│   └── plugins/                      ← Plugin modules (auto-discovered)
│       ├── __init__.py               ← SMKPlugin base class
│       ├── market_rhythm.py          ← MarketRhythmPlugin (MIR, spectrogram, λ3 harmony)
│       ├── market_heuristic.py       ← MarketHeuristicPlugin (AV-style wick/vol/entropy)
│       ├── market_vision.py          ← MarketVisionPlugin (SIFT keypoints + Smith-Waterman)
│       ├── market_seismology.py      ← MarketSeismologyPlugin (P/S/Surface wave)
│       └── kali_forensics.py         ← FileCarvingPlugin + SignatureScanPlugin
│
├── core/                             ← L1 SMK modules
│   ├── bias_detector.py
│   ├── dealing_range_detector.py
│   ├── equilibrium_cross_detector.py
│   ├── ipda_phase_detector.py
│   ├── premium_discount_detector.py
│   ├── session_detector.py
│   └── swing_detector.py
│
├── lambda_sensors/                   ← L3 SMK modules
│   ├── displacement_detector.py
│   ├── expansion_predictor.py
│   ├── harmonic_trap_detector.py
│   ├── manipulation_detector.py
│   └── volatility_decay_detector.py
│
├── liquidity/                        ← L2 SMK modules
│   ├── fvg_detector_engine.py
│   ├── order_block_detector.py
│   └── volume_profile_memory_engine.py
│
├── detectors/                        ← L4 SMK modules
│   ├── kl_divergence_detector.py
│   └── topological_fracture_detector.py
│
├── risk/                             ← Ring 0 modules
│   └── mandra_kernels.py
│
├── causality/                        ← Causal inference engines (not wired into main pipeline yet)
│   ├── granger_causality.py
│   ├── transfer_entropy.py
│   ├── ccm_engine.py
│   ├── spearman_lag_engine.py
│   └── signal_fusion_kernel.py
│
├── aegis-signals/                    ← AEGIS signal modules (new, not yet fully integrated)
├── frontend/
│   └── index.html                    ← Complete trading terminal (single-file, 1384 lines)
└── logs/                             ← Auto-created: events.log, veto.log, trades.log, session.log, raw_bars.log
```

---

## 3. TECHNOLOGY STACK

| Layer | Tech |
|---|---|
| Backend runtime | Python 3.12–3.13 + FastAPI + Uvicorn ASGI |
| Signal processing | NumPy, SciPy, JAX/XLA (GPU-accelerated on Roberto's machine) |
| Statistical | statsmodels VAR, scipy.signal FFT, sklearn DBSCAN |
| Topology | ripser (Vietoris-Rips persistent homology) |
| Chart rendering | TradingView Lightweight Charts v4 (CDN, WebGL) |
| Data transport | FastAPI WebSocket — bar-by-bar streaming |
| Live feed | Bitget REST API polling (5s) |
| Logging | Python RotatingFileHandler — 5 log streams |
| Frontend | Single HTML file — no build step, no framework |

---

## 4. SYSTEM ARCHITECTURE

### Data Flow

```
Data Source (CSV / Bitget / Sample)
    ↓
data_connectors.py  →  normalized bars [{time, open, high, low, close, volume}]
    ↓
SMKPipeline.load_bars()
    ↓  [on every bar via step()]
┌─────────────────────────────────────────────────────┐
│  L1 Structural Compiler (6 modules)                  │
│  L2 Memory & Imbalance (4 modules)                   │
│  L3 Dynamic Sensors (5 modules)                      │
│  L4 Ring 0 Veto (3 modules)                          │
│  Plugin Layer (6 plugins, appended to sensors[])     │
│  AEGIS Execution Bridge                              │
└─────────────────────────────────────────────────────┘
    ↓  bar result dict
WebSocket /ws/stream  →  frontend handleBar()
    ↓
TradingView chart + sensor panel + execution panel + logs
```

### AMD State Machine

```
Accumulation → Manipulation → Distribution → Retracement → Accumulation
R_MASTER = KL_fracture AND Topology_fracture → forced reset to Accumulation
```

### Ring 0 Veto Conditions (ANY = HALT, u_t = 0)

```
MANDRA:DE<0          Mandra Gate: negative information energy gain
TOPO:H1_FRACTURE     Persistent homology loop sum > threshold
FUSION:LAMBDA_VETO   Any lambda sensor issued hard veto
L3:LIAR_STATE        FFT phase inversion > π/2 (harmonic trap)
KL:REGIME_FRACTURE   KL divergence > 1.3× threshold
CONF:INSUFFICIENT    Fusion confidence < 0.2
```

---

## 5. BACKEND — KEY FILES

### `main.py` (424 lines)

FastAPI application. All imports are lazy (inside functions) to prevent crash on missing deps.

**REST Endpoints:**
```
POST /api/load/csv           {text, filename, source_hint}  → loads CSV into pipeline
POST /api/load/bitget        {api_key, api_secret, symbol, granularity, limit}
POST /api/load/oanda         {token, account_id, instrument, granularity, count}
POST /api/load/sample        → loads 300-bar synthetic EURUSD AMD cycle
POST /api/config/modules     {disabled_modules: [list]}  → toggle 18 SMK modules
POST /api/live/start         {api_key, symbol, granularity}  → start Bitget polling
POST /api/live/stop
GET  /api/live/status
GET  /api/status
GET  /api/ping
GET  /api/logs               → list log files with sizes
GET  /api/logs/{filename}    ?lines=100  → read last N lines
GET  /api/plugins            → list plugin status
POST /api/plugins/toggle     {enabled: [list of plugin names]}
GET  /api/execution/status   → AEGIS bridge availability
GET  /api/execution/stats    → StopLossManager session stats
POST /api/execution/configure {capital, risk_per_trade, n_venues, kelly_limit, enabled}
```

**WebSocket Endpoints:**
```
WS /ws/stream   ← backtest streaming
  Client sends: {action: "run"|"step"|"stop"|"reset", speed: int_ms}
  Server sends: {type: "bar", data: {full SMK result dict}}
              | {type: "done"}
              | {type: "stopped"}
              | {type: "error", message: str}

WS /ws/live     ← live Bitget feed
  Client sends: "ping"
  Server sends: {type: "bar", data: {full SMK result dict, live: true, symbol: str}}
```

**JSON Safety:** `_SafeEncoder` class handles all numpy scalars via `.item()`. All step() output passes through `_sanitize()` in smk_pipeline.py which recursively converts numpy types.

### `smk_pipeline.py` (674 lines)

Core pipeline. One `SMKPipeline` singleton per server instance.

**Key methods:**
```python
pipeline.load_bars(bars)      # resets cursor, warms plugins + ATR
pipeline.step()               # processes one bar, returns full result dict
pipeline.reset_cursor()       # reset to bar 0 without reloading data
pipeline.get_status()         # bars_loaded, cursor, amd_state, modules_ok/failed
```

**step() result dict keys:**
```
bar, bar_index, total_bars
dealing_range  {high, low, eq, zone, coherence, status}
bias           {bias, eq, zone, coherence, valid}
ipda_phase     {phase, eq, confidence, valid}
eq_cross       {zone, cross, direction, confidence}
session        {active, name, killzone, score, status}
swings         {count, nodes[{idx, price, type}]}
fvg            {count, active, recent[{type, top, bot, eq}]}
ob             {count, active, recent[{type, level, high, low, score}]}
vol_profile    {zones, hvn[{price, density}]}
vol_decay      {ratio, entrapped, energy, stasis, status}
displacement   {is_disp, dir, body_ratio, range_mult, vetoed, status}
harmonic       {phase_diff, inverted, freq, trap, status}
expansion      {sigma, prob, entrapped, target, status}
manipulation   {active, score, level, wick, status}
kl             {score, stable, h_curr, h_ref, status}
topology       {h1_score, fractured, islands, status}
amd            {state, prev, changed, R_MASTER}
fusion         {p_fused, confidence, veto_active, active_lambdas, regime, status}
mandra         {open, delta_e, size, regime_stable, status}
veto           {decision, reasons[], trade_allowed}
sensors        [{id, name, score, active, layer, status}]  ← 14 SMK + 6 plugin rows
plugins        {MarketRhythm: {...}, MarketHeuristic: {...}, ...}
execution      {action, reason, is_armed, pattern, dominant, direction,
                stop_loss_price, take_profit_price, lot_size, kelly_size,
                venue_allocation[], risk_pips, rr_ratio, delta_e, rev_score,
                risk_profile}
```

**Module loading:** All 18 modules are loaded with `try_load()` — failures log to `_import_errors` but don't crash. Numpy fallbacks cover every module. Real modules require `SMK_DIR` env var pointing to the project root (set by `start.bat`).

**Path resolution:** `_find_smk_root()` checks env var `SMK_DIR` first, then parent of `backend/`, then common sibling names.

**Plugin hook:** After all 18 SMK detectors run, `_get_plugins()` calls `plugin_manager.run()` which appends plugin sensor rows to `r['sensors']` and stores full results in `r['plugins']`.

**AEGIS hook:** After plugins, `_get_bridge()` calls `AegisBridge.evaluate()` on PROCEED bars. Result stored in `r['execution']`.

### `data_connectors.py`

Parses OHLCV from any source. Every loader calls `_normalize_bars()` which guarantees `time` as plain int unix seconds, deduplicates, and sorts.

**Supported CSV formats:**
```
MT4:        <DATE>,<TIME>,<OPEN>,...    2024.01.15,00:00
MT5:        DATE,TIME,OPEN,...          2024.01.15,00:00:00  (with TICKVOL)
TradingView ISO: time,open,...          2024-01-15T00:00:00+00:00
TradingView US:  Date,Time,Open,...     01/15/2024,00:00
Dukascopy:  UTC,Open,...               19.03.2026 12:00:00.000 UTC
Generic:    any delimiter, auto-detect columns
```

**`source_hint` parameter:** `'mt4'|'mt5'|'tradingview'|'dukascopy'|'auto'` — passed from frontend DATA SOURCE card selection.

### `logger.py`

Five rotating log streams (10MB each, 5 backups):
```
logs/events.log    ← FVGs, AMD transitions, Judas swings, signals
logs/veto.log      ← every bar's Ring 0 decision
logs/trades.log    ← trade opens/closes
logs/session.log   ← server start, data loads, live feed
logs/raw_bars.log  ← full JSON per bar (50MB limit)
```

Read via API: `GET /api/logs/events.log?lines=200`

### `realtime.py` — Live Feed

Polls Bitget REST `/api/v2/spot/market/candles` every 5 seconds. Bootstraps 100 bars of history on start to warm the pipeline. New closed candles run through `pipeline.step()` and broadcast to `/ws/live` subscribers.

```python
live_feed.configure(symbol, granularity, api_key, pipeline)
live_feed.start(pipeline)   # creates asyncio.Task
live_feed.stop()
```

---

## 6. PLUGIN SYSTEM

### Plugin Base Class (`plugins/__init__.py`)

```python
class SMKPlugin(ABC):
    name:            str    # displayed in settings panel
    layer:           str    # "L1"|"L2"|"L3"|"L4"|"λ-ext"|"EXE"
    sensor_id:       str    # "p01".."p06" etc
    requires_warmup: int    # bars before plugin fires (default 20)

    @abstractmethod
    def update(self, bar: dict, df: pd.DataFrame, smk: dict) -> dict:
        # Returns flat dict — ALL values must be plain Python types
        # Must always contain: "active" (bool), "score" (float 0-1), "status" (str)
        ...
```

### Adding a New Plugin

1. Create `backend/plugins/my_plugin.py` with a class extending `SMKPlugin`
2. Add entry to `PLUGIN_REGISTRY` in `plugin_manager.py`:
   ```python
   ("plugins.my_plugin", "MyPluginClass"),
   ```
3. Restart server — plugin auto-loads

### Current Plugins

| Sensor ID | Name | Layer | What it detects |
|---|---|---|---|
| p01 | MarketRhythm | L3-ext | Spectrogram tempo, λ3 harmony, Shazam fingerprint |
| p02 | MarketHeuristic | L3-ext | Stop-hunt wick, volume spike, entropy threat score |
| p03 | MarketVision | L2-ext | SIFT keypoints, ORB matching, Smith-Waterman homology |
| p04 | MarketSeismology | L3-ext | P/S/Surface wave, phase entrapment, expansion critical |
| p05 | FileCarvingEngine | L2-ext | S/R level clustering, accumulation zone carving |
| p06 | SignatureScanEngine | L2-ext | Entropy regime, 14-pattern signature DB, CLM tokenization |

**Deferred plugins (not yet built):**
- `RadarPulseDetector` — MAD clutter, PRI rhythm, matched filter (needs tick data or OHLCV derivation)
- `PriceUVMapper` — Vietoris-Rips UV unwrap, Delaunay heatmap rendering
- `CandleLanguageModel` — sentence_transformers contextual embedding

---

## 7. AEGIS EXECUTION BRIDGE

### `aegis_bridge.py` (379 lines)

Wires SMK output into the full execution chain. Lazy-imports both `StopLossManager` and `AegisExtensions` — missing deps don't crash the bridge.

**Signal chain on every PROCEED bar:**
```
CLMTokenizer.sequence(last 8 bars)     → "BWBUI"
SequenceStopLossManager.calculate_from_sequence()
    dominant symbol → SL%, TP mult, lot size, R:R gate
AegisExtensions.on_signal()  (if installed)
    MandraBitGate → RevPeriodGuard → SchurRouter
    → kelly_size + venue_allocation[]
```

**CLM 7-symbol alphabet:**
```
B = Strong Bullish (body/range > 0.6, close > open)
I = Strong Bearish (body/range > 0.6, close < open)
W = Upper Wick reversal (upper_wick > 2× body, close < open)
w = Lower Wick reversal (lower_wick > 2× body, close > open)
U = Weak Bull (body/range 0.1–0.6, close > open)
D = Weak Bear (body/range 0.1–0.6, close < open)
X = Neutral / structure (body/range < 0.1)
```

**Dominance hierarchy for sequence resolution:**
```
W/w = 5  (highest — wick reversals dominate)
B/I = 4
X   = 3
U/D = 2
```

**`execution` dict in bar result:**
```python
{
    "action":            "TRADE"|"HALT"|"REDUCE"|"WARMUP",
    "reason":            str,
    "is_armed":          bool,
    "pattern":           str,        # e.g. "BWBUI"
    "dominant":          str,        # e.g. "W"
    "direction":         int,        # 1=Long, -1=Short, 0=None
    "stop_loss_price":   float,
    "take_profit_price": float,
    "lot_size":          float,
    "kelly_size":        float,      # fraction of capital
    "venue_allocation":  list[float],# per-venue fractions
    "risk_pips":         float,
    "rr_ratio":          float,
    "delta_e":           float,      # Mandra ΔE
    "rev_score":         float,      # reverse period score
    "risk_profile":      str,        # full summary string
}
```

### Files That Must Be Present for Full Execution

```
smk_ipda_trading_system/
├── aegis_extensions.py         ← project root (SchurRouter, MandraBitGate, etc.)
└── backend/
    └── stop_loss_manager.py    ← MOVE here from project root
```

Both files are auto-discovered by `aegis_bridge.py` via `sys.path` which includes both `backend/` and the project root.

---

## 8. FRONTEND (`frontend/index.html`)

Single 1384-line HTML file. No build step. No framework. Opens directly in browser OR served by FastAPI at `/`.

**API base detection:**
```javascript
const API_BASE = (window.location.protocol === 'file:')
  ? 'http://localhost:8000'
  : window.location.origin;
const WS_BASE = API_BASE.replace(/^http/, 'ws');
```

### Layout

```
┌─ TITLEBAR ──────────────────────────────────────────────────────────────┐
│  QUIMERIA SMK   [conn pill] [src pill] [AMD pill] [signal pill] [dot]   │
├─ CTRL BAR ──────────────────────────────────────────────────────────────┤
│  [DATA SOURCE] [⚙ SETTINGS] [◉ LIVE] [SAMPLE] [▶ RUN] [■ STOP] [STEP] │
│  [RESET] speed▾  ████progress████  filename                             │
├─ LEFT ────────┬─ CENTER (CHART) ────────────────┬─ RIGHT ───────────────┤
│ 14 SMK sensor │ TradingView candlestick chart    │ EXECUTION ENGINE      │
│ bars          │ + volume histogram               │ VETO AUTHORITY        │
│ + 6 plugin    │ + reversal markers               │ AMD STATE             │
│ sensor bars   │ + DR/EQ price lines              │ FUSED SIGNAL          │
│               │ + SL/TP dashed lines             │ DEALING RANGE         │
│ CAUSAL LAYER  │                                  │ VOLATILITY DECAY      │
│               ├─ POSITION BAR ───────────────────┤ SPECTRAL STATUS       │
│               │ open positions + P&L             │ MANDRA GATE           │
│               ├─ TRADE SIMULATOR ────────────────┤ MANIPULATION          │
│               │ lot size | BUY | SELL | CLOSE ALL│                       │
├─ LOG PANELS ──┴──────────────────────────────────┴───────────────────────┤
│  EVENT LOG  │  VETO STREAM  │  TRADE LOG                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key JavaScript Functions

```javascript
connectWS(cb)           // open /ws/stream if not already open
wsSend(obj)             // send action to WS, auto-reconnect if needed
apiLoad(path, body)     // POST to /api/load/{path}
handleBar(d)            // process bar result → update all panels
detectAndPaintReversal(d)  // add markers to chart on AMD transitions
updateTradeSignals(d)   // BUY/SELL button glow based on veto+fusion
updateExecution(exe)    // update execution panel + SL/TP chart lines
startLive()             // start Bitget live feed + connect /ws/live
stopLive()              // stop live feed
buildModuleToggles()    // populate settings modal with 18 SMK modules
applySettings()         // POST /api/config/modules with disabled list
```

### Reversal Markers on Chart

```
green arrow below bar  = Manipulation→Distribution (BUY reversal)
red arrow above bar    = Distribution→Retracement (SELL/close reversal)
orange circle          = Judas Swing / manipulation detected
blue arrow             = λ1 critical mass (entrapment + stasis)
orange square          = HALT (topology fracture or KL break)
grey circle            = R_MASTER reset
```

### Modals

- **DATA SOURCE** — 7 cards: MT4, MT5, TradingView, Generic CSV, Bitget, OANDA, Built-in Sample
- **⚙ SETTINGS** — 18 module toggles (color-coded L1=green, λ=orange, L2=grey, L4=red), speed, lookback, line visibility
- **◉ LIVE** — Bitget symbol/TF/API key → connects `/ws/live`

---

## 9. DEPLOYMENT

### Roberto's Machine (Windows)

```
C:\Users\Roberto\Documents\chimeria\smk_ipda_trading_system\
```

Start command:
```cmd
cd C:\...\smk_ipda_trading_system\backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or double-click `start.bat` from project root.

### Production Server (Linux — mt.itimbre.com)

```
/home/rjimenez/chimeria/smk_ipda_trading_system/
```

```bash
cd ~/chimeria/smk_ipda_trading_system
./start.sh
```

Access: `http://mt.itimbre.com:8000` (port 8000 is open in firewall, port 80 blocked at Proxmox VM level — Apache reverse proxy config exists but Proxmox firewall rule not yet added).

### Environment Variables

```bash
SMK_DIR=/path/to/smk_ipda_trading_system   # tells smk_pipeline.py where modules are
PYTHONPATH=$SMK_DIR:$PYTHONPATH              # set by start.sh
```

---

## 10. KNOWN ISSUES / TECHNICAL DEBT

| Issue | Status | Notes |
|---|---|---|
| `stop_loss_manager.py` is at project root | Needs to move to `backend/` | `aegis_bridge.py` finds it via sys.path but cleaner in `backend/` |
| Port 80 blocked | Proxmox VM firewall | Add inbound rule for port 80 at Proxmox host level, then Apache reverse proxy works |
| `smk_pipeline.py` disk corruption | Solved with `fix_pipeline.py` | Windows download/copy can corrupt triple-quote strings. Run `python fix_pipeline.py` from backend/ |
| Numpy type serialization | Solved | `_sanitize()` in smk_pipeline + `_SafeEncoder` in main.py cover all numpy generics |
| WebSocket 2-candle stop | Solved | run_loop is asyncio.Task; connectWS only creates new socket if not already open; never reset_cursor on new connection |
| AEGIS async in sync context | Partial | `aegis_bridge.py` handles running/non-running event loop via ThreadPoolExecutor fallback |
| CandleLanguageModel (sentence_transformers) | Deferred | Tokenizer works without transformer; embedding deferred to later |
| RadarPulseDetector (tick data) | Deferred | Needs tick intervals; derivable from OHLCV but not yet implemented |
| PriceUVMapper rendering | Deferred | Telemetry computed, frontend heatmap not yet built |
| Causality engines | Not wired | `causality/` folder has Granger/TE/CCM/Spearman engines but not connected to main pipeline |
| HYPERION integration | Planned | Rust execution engine not yet connected; AEGIS bridge is the connection point |

---

## 11. WHAT HYPERION IS (context for integration)

HYPERION is a separate Rust+Python+React system that handles:
- High-performance tick processing (Rust/PyO3)
- Real order execution via Bitget API
- Schur-complement venue routing (61.8% dark / 38.2% lit)
- React dashboard

**Integration plan:** QUIMERIA's `execution.action == "TRADE"` + `execution.lot_size` + `execution.venue_allocation` → HYPERION's `engine.execute_trade(signal, size)`. The AEGIS bridge is the connection point. Currently both systems run independently.

---

## 12. DEPENDENCIES

### Python (backend)

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
httpx>=0.27.0
numpy>=1.24
pandas>=2.0
scipy>=1.10
statsmodels>=0.14       (has shim if missing)
scikit-learn>=1.3
jax>=0.4.0              (has shim if missing — uses numpy fallback)
jaxlib>=0.4.0
ripser>=0.6.0           (for topological_fracture_detector)
pytz
python-multipart
websockets              (for future WS client)
```

### Frontend

```
TradingView Lightweight Charts v4.1.3  (CDN: unpkg.com)
```
No other JS dependencies. No build step.

---

## 13. CODING CONVENTIONS

**Python:**
- All module telemetry fields explicitly cast at extraction: `float(t.score)`, `bool(t.is_valid)`, `str(t.status)`
- Never return numpy scalars from step() — `_sanitize()` is the safety net but explicit casting is preferred
- All new modules follow the `@dataclass` telemetry pattern
- Fallback logic for every detector — server never crashes on module failure
- Lazy imports in all FastAPI endpoints — missing deps cause 500 on that endpoint only, not server crash

**JavaScript:**
- `wsSend()` handles reconnection — never call `connectWS()` directly on button clicks
- All chart operations wrapped in `try/catch` — LWC throws on invalid time series
- `handleBar()` is the single entry point for all bar data — both backtest and live
- `setText(id, val, color)` and `setPill(id, txt, color)` are the UI update helpers

**File safety:**
- `smk_pipeline.py` gets corrupted on Windows download — always use `fix_pipeline.py` to write it
- Never use heredoc strings (`<< 'EOF'`) in bash scripts that contain triple-quotes — use Python `create_file` or write via `open()`

---

## 14. QUICK REFERENCE — HOW TO ADD THINGS

### New SMK detector module
1. Add Python file in `core/`, `lambda_sensors/`, `liquidity/`, or `detectors/`
2. Add `try_load("key", lambda: _imp("module.path", "ClassName")())` in `SMKPipeline._load_modules()`
3. Add `self._mymodule(df)` call in `step()` and implement `_mymodule()` method with fallback

### New plugin
1. Create `backend/plugins/my_plugin.py` extending `SMKPlugin`
2. Add `("plugins.my_plugin", "MyClass")` to `PLUGIN_REGISTRY` in `plugin_manager.py`
3. Plugin auto-discovers on restart

### New REST endpoint
Add to `main.py` before the websocket handler. Use `from module import X` inside the function body (lazy import pattern).

### New frontend panel
1. Add HTML in the appropriate panel div
2. Add JS update function called from `handleBar(d)`
3. Clear state in `resetLocal()`

### New data source
1. Add parser function in `data_connectors.py` returning `List[Dict]`
2. Call `_normalize_bars()` before returning
3. Add card in the DATA SOURCE modal HTML
4. Add case to `pickSrc()` in frontend JS

---

## 15. SESSION LOG (last known working state)

```
[2026-04-21] Server starts cleanly on Windows (Python 3.13)
[2026-04-21] All 18 SMK modules load with JAX available
[2026-04-21] 6 plugins load and run (warmup then active)
[2026-04-21] CSV import works for Dukascopy EUR/USD 1M data (719 bars)
[2026-04-21] Backtest streams bars continuously (2-candle bug fixed)
[2026-04-21] Reversal markers paint on chart
[2026-04-21] BUY/SELL buttons glow on signal
[2026-04-21] AEGIS bridge wired — awaits stop_loss_manager.py in backend/
[2026-04-21] Logs writing to smk_ipda_trading_system/logs/
[2026-04-21] Demo URL: http://mt.itimbre.com:8000
```
