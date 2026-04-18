"""
Data Connectors — CSV, Bitget REST, OANDA REST, synthetic sample
"""
import csv, io, math, random, time
from typing import List, Dict, Optional
import httpx


# ─── CSV ─────────────────────────────────────────────────────────────────────
def load_csv_text(text: str) -> Optional[List[Dict]]:
    """
    Parse OHLCV CSV — handles MT4, TradingView, generic formats.
    Auto-detects delimiter and column names.
    """
    text = text.strip()
    if not text:
        return None

    # Detect delimiter
    for delim in [",", "\t", ";"]:
        if delim in text.split("\n")[0]:
            break
    else:
        delim = ","

    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    headers = [h.strip().lower().replace('"', '').replace("'", '') for h in (reader.fieldnames or [])]

    def find(names):
        for n in names:
            for i, h in enumerate(headers):
                if n in h:
                    return reader.fieldnames[i]
        return None

    time_col  = find(["datetime", "date", "time", "timestamp"])
    open_col  = find(["open"])
    high_col  = find(["high"])
    low_col   = find(["low"])
    close_col = find(["close"])
    vol_col   = find(["volume", "vol", "tick"])

    if not all([open_col, high_col, low_col, close_col]):
        return None

    bars = []
    for row in reader:
        try:
            ts = int(time.mktime(time.strptime(
                row[time_col].strip().replace('"',''),
                _detect_date_fmt(row[time_col].strip())
            ))) if time_col and row.get(time_col) else int(time.time()) + len(bars) * 300
        except:
            ts = int(time.time()) + len(bars) * 300

        try:
            o = float(row[open_col]);  h = float(row[high_col])
            l = float(row[low_col]);   c = float(row[close_col])
            v = float(row[vol_col]) if vol_col and row.get(vol_col) else 100.0
        except:
            continue

        if any(math.isnan(x) for x in [o, h, l, c]):
            continue

        bars.append({"time": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})

    bars.sort(key=lambda b: b["time"])
    return _normalize_bars(bars) if len(bars) >= 3 else None


def _detect_date_fmt(s: str) -> str:
    s = s.strip().replace('"', '').replace("'", '')
    if "T" in s:   return "%Y-%m-%dT%H:%M:%S"
    if "/" in s:
        if s.count("/") == 2:
            parts = s.split("/")
            if len(parts[0]) == 4: return "%Y/%m/%d %H:%M"
            return "%m/%d/%Y %H:%M"
    if " " in s:   return "%Y.%m.%d %H:%M"
    return "%Y-%m-%d %H:%M:%S"


# ─── BITGET ──────────────────────────────────────────────────────────────────
BITGET_GRAN_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1H", "4h": "4H", "1d": "1D"
}

async def fetch_bitget(api_key: str, api_secret: str, symbol: str,
                       granularity: str = "5m", limit: int = 300) -> List[Dict]:
    """
    Fetch historical klines from Bitget v2 REST API.
    Docs: https://www.bitget.com/api-doc/spot/market/Get-Candlestick-Data
    """
    gran = BITGET_GRAN_MAP.get(granularity.lower(), granularity)
    url = "https://api.bitget.com/api/v2/spot/market/candles"
    params = {"symbol": symbol.upper(), "granularity": gran, "limit": str(min(limit, 1000))}

    headers = {
        "ACCESS-KEY": api_key,
        "Content-Type": "application/json",
    }

    # Bitget public candlestick endpoint does not require signature for market data
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    if data.get("code") != "00000":
        raise ValueError(f"Bitget error: {data.get('msg', 'unknown')}")

    bars = []
    for candle in data.get("data", []):
        # [timestamp_ms, open, high, low, close, base_vol, quote_vol]
        ts  = int(candle[0]) // 1000
        o   = float(candle[1]); h = float(candle[2])
        l   = float(candle[3]); c = float(candle[4])
        v   = float(candle[5])
        bars.append({"time": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})

    return _normalize_bars(bars)


# ─── OANDA ───────────────────────────────────────────────────────────────────
OANDA_GRAN_MAP = {
    "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
    "1h": "H1", "4h": "H4", "1d": "D"
}

async def fetch_oanda(token: str, account_id: str, instrument: str = "EUR_USD",
                      granularity: str = "M5", count: int = 300) -> List[Dict]:
    """
    Fetch candles from OANDA v20 REST API.
    Works with both fxTrade live and Practice accounts.
    """
    gran = OANDA_GRAN_MAP.get(granularity.lower(), granularity)

    # Try live first, fall back to practice
    for host in ["https://api-fxtrade.oanda.com", "https://api-fxpractice.oanda.com"]:
        url = f"{host}/v3/instruments/{instrument}/candles"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        params  = {"count": str(min(count, 5000)), "granularity": gran, "price": "M"}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, params=params, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    bars = []
                    for c in data.get("candles", []):
                        if not c.get("complete", True):
                            continue
                        mid = c.get("mid", {})
                        ts  = int(time.mktime(time.strptime(c["time"][:19], "%Y-%m-%dT%H:%M:%S")))
                        bars.append({
                            "time": ts,
                            "open":   float(mid.get("o", 0)),
                            "high":   float(mid.get("h", 0)),
                            "low":    float(mid.get("l", 0)),
                            "close":  float(mid.get("c", 0)),
                            "volume": float(c.get("volume", 100)),
                        })
                    bars.sort(key=lambda b: b["time"])
                    return _normalize_bars(bars)
        except Exception:
            continue

    raise ValueError("OANDA connection failed — check token and account type")


# ─── SAMPLE GENERATOR ────────────────────────────────────────────────────────
def generate_sample(n: int = 300, instrument: str = "EURUSD", tf_seconds: int = 300) -> List[Dict]:
    """
    Synthetic OHLCV with embedded ACC → MAN → DIS → RET AMD cycle.
    Includes FVG events, OB candles, harmonic trap, manipulation wick.
    """
    bars = []
    p = 1.1050
    # Start 24h ago
    t = int(time.time()) - n * tf_seconds

    for i in range(n):
        # Phase envelope
        phase = 0 if i < n*0.25 else 1 if i < n*0.45 else 2 if i < n*0.75 else 3
        noise_scale = [0.0008, 0.0015, 0.0025, 0.0012][phase]
        trend_bias  = [0.0, 0.0003 * math.sin(i * 0.3), 0.0012, -0.0005][phase]

        o = p
        body = random.gauss(trend_bias, noise_scale)
        c = o + body

        wick_h = random.expovariate(1 / 0.0004)
        wick_l = random.expovariate(1 / 0.0004)

        # Manipulation wick at phase boundary
        if i == int(n * 0.44):
            wick_h = 0.0025  # Judas sweep up

        h = max(o, c) + wick_h
        l = min(o, c) - wick_l

        # Volume: spike on phase transitions
        base_vol = 400 + random.expovariate(1/300)
        if phase == 2: base_vol *= 2.2
        if i in [int(n*0.25), int(n*0.45)]: base_vol *= 3.5

        bars.append({
            "time":   t + i * tf_seconds,
            "open":   round(o, 5),
            "high":   round(h, 5),
            "low":    round(l, 5),
            "close":  round(c, 5),
            "volume": int(base_vol),
        })
        p = c

    return bars


def _normalize_bars(bars):
    """Ensure every bar has time as int unix seconds and standard OHLCV keys."""
    import time as _time
    out = []
    for b in bars:
        # Resolve time field
        t = b.get('time') or b.get('timestamp') or b.get('datetime') or b.get('date')
        if isinstance(t, str):
            try:
                import datetime
                for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S',
                            '%Y-%m-%dT%H:%M:%SZ', '%Y.%m.%d %H:%M',
                            '%Y-%m-%d', '%m/%d/%Y %H:%M'):
                    try:
                        dt = datetime.datetime.strptime(t[:19], fmt)
                        t = int(dt.timestamp())
                        break
                    except ValueError:
                        continue
                else:
                    t = int(_time.time())
            except Exception:
                t = int(_time.time())
        elif isinstance(t, (int, float)):
            # If milliseconds (>1e10), convert to seconds
            t = int(t / 1000) if t > 1e10 else int(t)
        else:
            t = int(_time.time())

        out.append({
            'time':   t,
            'open':   float(b.get('open', 0)),
            'high':   float(b.get('high', 0)),
            'low':    float(b.get('low', 0)),
            'close':  float(b.get('close', 0)),
            'volume': float(b.get('volume', 100)),
        })
    # Sort by time ascending and deduplicate
    out.sort(key=lambda x: x['time'])
    seen = set()
    deduped = []
    for bar in out:
        if bar['time'] not in seen:
            seen.add(bar['time'])
            deduped.append(bar)
    return deduped"""
Data Connectors — CSV, Bitget REST, OANDA REST, synthetic sample
"""
import csv, io, math, random, time
from typing import List, Dict, Optional
import httpx


# ─── CSV ─────────────────────────────────────────────────────────────────────
def load_csv_text(text: str) -> Optional[List[Dict]]:
    """
    Parse OHLCV CSV — handles MT4, TradingView, generic formats.
    Auto-detects delimiter and column names.
    """
    text = text.strip()
    if not text:
        return None

    # Detect delimiter
    for delim in [",", "\t", ";"]:
        if delim in text.split("\n")[0]:
            break
    else:
        delim = ","

    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    headers = [h.strip().lower().replace('"', '').replace("'", '') for h in (reader.fieldnames or [])]

    def find(names):
        for n in names:
            for i, h in enumerate(headers):
                if n in h:
                    return reader.fieldnames[i]
        return None

    time_col  = find(["datetime", "date", "time", "timestamp"])
    open_col  = find(["open"])
    high_col  = find(["high"])
    low_col   = find(["low"])
    close_col = find(["close"])
    vol_col   = find(["volume", "vol", "tick"])

    if not all([open_col, high_col, low_col, close_col]):
        return None

    bars = []
    for row in reader:
        try:
            ts = int(time.mktime(time.strptime(
                row[time_col].strip().replace('"',''),
                _detect_date_fmt(row[time_col].strip())
            ))) if time_col and row.get(time_col) else int(time.time()) + len(bars) * 300
        except:
            ts = int(time.time()) + len(bars) * 300

        try:
            o = float(row[open_col]);  h = float(row[high_col])
            l = float(row[low_col]);   c = float(row[close_col])
            v = float(row[vol_col]) if vol_col and row.get(vol_col) else 100.0
        except:
            continue

        if any(math.isnan(x) for x in [o, h, l, c]):
            continue

        bars.append({"time": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})

    bars.sort(key=lambda b: b["time"])
    return bars if len(bars) >= 3 else None


def _detect_date_fmt(s: str) -> str:
    s = s.strip().replace('"', '').replace("'", '')
    if "T" in s:   return "%Y-%m-%dT%H:%M:%S"
    if "/" in s:
        if s.count("/") == 2:
            parts = s.split("/")
            if len(parts[0]) == 4: return "%Y/%m/%d %H:%M"
            return "%m/%d/%Y %H:%M"
    if " " in s:   return "%Y.%m.%d %H:%M"
    return "%Y-%m-%d %H:%M:%S"


# ─── BITGET ──────────────────────────────────────────────────────────────────
BITGET_GRAN_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1H", "4h": "4H", "1d": "1D"
}

async def fetch_bitget(api_key: str, api_secret: str, symbol: str,
                       granularity: str = "5m", limit: int = 300) -> List[Dict]:
    """
    Fetch historical klines from Bitget v2 REST API.
    Docs: https://www.bitget.com/api-doc/spot/market/Get-Candlestick-Data
    """
    gran = BITGET_GRAN_MAP.get(granularity.lower(), granularity)
    url = "https://api.bitget.com/api/v2/spot/market/candles"
    params = {"symbol": symbol.upper(), "granularity": gran, "limit": str(min(limit, 1000))}

    headers = {
        "ACCESS-KEY": api_key,
        "Content-Type": "application/json",
    }

    # Bitget public candlestick endpoint does not require signature for market data
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    if data.get("code") != "00000":
        raise ValueError(f"Bitget error: {data.get('msg', 'unknown')}")

    bars = []
    for candle in data.get("data", []):
        # [timestamp_ms, open, high, low, close, base_vol, quote_vol]
        ts  = int(candle[0]) // 1000
        o   = float(candle[1]); h = float(candle[2])
        l   = float(candle[3]); c = float(candle[4])
        v   = float(candle[5])
        bars.append({"time": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})

    bars.sort(key=lambda b: b["time"])
    return bars


# ─── OANDA ───────────────────────────────────────────────────────────────────
OANDA_GRAN_MAP = {
    "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
    "1h": "H1", "4h": "H4", "1d": "D"
}

async def fetch_oanda(token: str, account_id: str, instrument: str = "EUR_USD",
                      granularity: str = "M5", count: int = 300) -> List[Dict]:
    """
    Fetch candles from OANDA v20 REST API.
    Works with both fxTrade live and Practice accounts.
    """
    gran = OANDA_GRAN_MAP.get(granularity.lower(), granularity)

    # Try live first, fall back to practice
    for host in ["https://api-fxtrade.oanda.com", "https://api-fxpractice.oanda.com"]:
        url = f"{host}/v3/instruments/{instrument}/candles"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        params  = {"count": str(min(count, 5000)), "granularity": gran, "price": "M"}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, params=params, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    bars = []
                    for c in data.get("candles", []):
                        if not c.get("complete", True):
                            continue
                        mid = c.get("mid", {})
                        ts  = int(time.mktime(time.strptime(c["time"][:19], "%Y-%m-%dT%H:%M:%S")))
                        bars.append({
                            "time": ts,
                            "open":   float(mid.get("o", 0)),
                            "high":   float(mid.get("h", 0)),
                            "low":    float(mid.get("l", 0)),
                            "close":  float(mid.get("c", 0)),
                            "volume": float(c.get("volume", 100)),
                        })
                    bars.sort(key=lambda b: b["time"])
                    return bars
        except Exception:
            continue

    raise ValueError("OANDA connection failed — check token and account type")


# ─── SAMPLE GENERATOR ────────────────────────────────────────────────────────
def generate_sample(n: int = 300, instrument: str = "EURUSD", tf_seconds: int = 300) -> List[Dict]:
    """
    Synthetic OHLCV with embedded ACC → MAN → DIS → RET AMD cycle.
    Includes FVG events, OB candles, harmonic trap, manipulation wick.
    """
    bars = []
    p = 1.1050
    # Start 24h ago
    t = int(time.time()) - n * tf_seconds

    for i in range(n):
        # Phase envelope
        phase = 0 if i < n*0.25 else 1 if i < n*0.45 else 2 if i < n*0.75 else 3
        noise_scale = [0.0008, 0.0015, 0.0025, 0.0012][phase]
        trend_bias  = [0.0, 0.0003 * math.sin(i * 0.3), 0.0012, -0.0005][phase]

        o = p
        body = random.gauss(trend_bias, noise_scale)
        c = o + body

        wick_h = random.expovariate(1 / 0.0004)
        wick_l = random.expovariate(1 / 0.0004)

        # Manipulation wick at phase boundary
        if i == int(n * 0.44):
            wick_h = 0.0025  # Judas sweep up

        h = max(o, c) + wick_h
        l = min(o, c) - wick_l

        # Volume: spike on phase transitions
        base_vol = 400 + random.expovariate(1/300)
        if phase == 2: base_vol *= 2.2
        if i in [int(n*0.25), int(n*0.45)]: base_vol *= 3.5

        bars.append({
            "time":   t + i * tf_seconds,
            "open":   round(o, 5),
            "high":   round(h, 5),
            "low":    round(l, 5),
            "close":  round(c, 5),
            "volume": int(base_vol),
        })
        p = c

    return bars
