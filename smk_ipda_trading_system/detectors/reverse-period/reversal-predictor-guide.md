# IPDA Forex Reversal Prediction System
## Setup & Usage Guide

---

## What This System Does

This ML system predicts **high-probability IPDA reversal windows** in Forex markets by:

1. Fetching daily OHLCV data for any forex pair via Yahoo Finance
2. Engineering 30+ IPDA-specific features from the ICT framework
3. Labeling historical reversal periods algorithmically
4. Training an **XGBoost classifier** using time-series cross-validation
5. Outputting reversal probabilities for each daily bar
6. Generating a 6-panel analysis chart

---

## Installation

```bash
pip install yfinance pandas numpy scikit-learn xgboost matplotlib seaborn
```

---

## Running the System

```bash
python ipda_reversal_predictor.py
```

By default it runs on **EURUSD** from 2018 to today. To change the pair, edit the CONFIG block at the top of the file:

```python
CONFIG = {
    "pair":     "GBPUSD=X",   # Change pair here
    "start_date": "2018-01-01",
    "reversal_threshold_pct": 0.8,  # Adjust sensitivity
    "reversal_fwd_window":    10,   # Days forward to confirm reversal
}
```

**Supported pairs:** `EURUSD=X`, `GBPUSD=X`, `USDJPY=X`, `AUDUSD=X`, `USDCAD=X`, `USDCHF=X`, `NZDUSD=X`

---

## Features Engineered (30+)

### IPDA Core Features
| Feature | Description |
|---|---|
| `ipda_20d_pos` | Price position within 20-day range (0=discount, 1=premium) |
| `ipda_40d_pos` | Price position within 40-day range |
| `ipda_60d_pos` | Price position within 60-day range |
| `breach_high_20d` | Price swept the 20-day high (stop hunt signal) |
| `breach_low_40d` | Price swept the 40-day low |
| `above_equil_60d` | Price above the 60-day equilibrium level |
| `dist_from_60d_high` | ATR-normalized distance from 60-day high |

### Market Structure Features
| Feature | Description |
|---|---|
| `mss_bearish` | Bearish Market Structure Shift detected |
| `mss_bullish` | Bullish Market Structure Shift detected |
| `bull_fvg` | Bullish Fair Value Gap present |
| `bear_fvg` | Bearish Fair Value Gap present |
| `swing_high` | Current bar is a swing high |
| `swing_low` | Current bar is a swing low |

### Momentum & Volatility
| Feature | Description |
|---|---|
| `rsi_14` | RSI(14) — overbought/oversold |
| `rsi_ob` | RSI ≥ 70 (premium/overbought) |
| `rsi_os` | RSI ≤ 30 (discount/oversold) |
| `momentum_5/10/20` | Price rate of change over 5/10/20 days |
| `atr_pct` | ATR as % of price (volatility regime) |

### Cycle & Timing
| Feature | Description |
|---|---|
| `near_quarterly_shift` | Within 5 days of a 63-day IPDA quarterly boundary |
| `quarter_cycle_pos` | Position within the current 63-day quarter |
| `is_monday` | Monday (often weekly liquidity grab day) |
| `confluence_20d` | Combined breach + MSS + FVG score |

---

## How the Reversal Label is Created

A bar is labeled **Reversal = 1** when:
- The **current 5-bar trend is established** (up or down)
- **Within the next 10 bars**, price reverses by at least **0.8%** in the opposite direction

This captures the IPDA logic: price sweeps liquidity at a range boundary, then reverses to fill imbalances.

You can tune sensitivity:
- Raise `reversal_threshold_pct` → fewer but stronger reversals labeled
- Lower `reversal_fwd_window` → tighter, faster reversal confirmation

---

## Output

### Console Output
```
──────────────────────────────────────────────
  Pair:                 EURUSD
  Latest Date:          2025-03-21
  Reversal Probability: 72.4%
  Signal:               ⚠️  HIGH PROBABILITY REVERSAL
──────────────────────────────────────────────

Recent 10 bars — Reversal Probabilities:
  2025-03-10   18.3%  |███               |
  2025-03-11   22.1%  |████              |
  2025-03-12   71.8%  |██████████████    | ← SIGNAL
  2025-03-13   68.2%  |█████████████     | ← SIGNAL
  ...
```

### Chart (ipda_reversal_analysis.png)
1. **Price + Reversal Windows** — orange shading where model predicts reversals
2. **Probability Over Time** — full probability series with threshold line
3. **ROC Curve** — model discrimination quality
4. **Confusion Matrix** — prediction accuracy breakdown
5. **Feature Importances** — which IPDA signals matter most
6. **Recent 120 Days** — IPDA 20/40/60 ranges overlaid with reversal signals

---

## Interpreting Results

| Probability | Interpretation |
|---|---|
| 0–35% | No reversal expected — trend likely continues |
| 35–55% | Caution zone — watch for confirming signals |
| 55–75% | High probability reversal window — reduce exposure |
| 75–100% | Very high probability — consider counter-trend setup |

### Confluence Checklist (use alongside model output)
When the model shows ≥55%, look for **3 or more** of these to confirm:
- [ ] Price has swept a 20/40/60-day high or low
- [ ] MSS (Market Structure Shift) visible on H4 or H1
- [ ] Fair Value Gap present in the reversal zone
- [ ] RSI overbought (>70) or oversold (<30)
- [ ] Near quarterly shift boundary (every ~63 trading days)
- [ ] Strong wick rejection candle at range boundary
- [ ] Session: London or NY open kill zone

---

## Important Notes

- **Daily bars only** — IPDA is a daily-timeframe framework. Do not run on H1/H4 data.
- **Retrain monthly** — Run the script monthly to incorporate recent data.
- **This is probabilistic** — No model guarantees outcomes. Always use stop losses.
- Backtested performance does not guarantee future results.

---

## Extending the System

**Add more pairs in a loop:**
```python
pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
for pair in pairs:
    CONFIG["pair"] = pair
    # ... run pipeline
```

**Add higher timeframe confirmation (Weekly bias):**
```python
df_weekly = yf.download(PAIR, interval="1wk", ...)
# Engineer weekly trend feature, merge on date into daily df
```

**Export signals to CSV:**
```python
signal_df = model_df[["reversal"]].copy()
signal_df["prob"] = final_model.predict_proba(model_df[FEATURE_COLS].values)[:, 1]
signal_df.to_csv("ipda_signals.csv")
```
