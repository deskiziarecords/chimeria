"""
data_connectors.py  —  QUIMERIA SMK backend data sources.
Every loader returns List[Dict] with guaranteed keys:
  time (int unix seconds), open, high, low, close, volume
"""
import csv
import io
import math
import random
import time as _time
import datetime
import re
from typing import List, Dict, Optional
import httpx


# ── TIME PARSER ───────────────────────────────────────────────────────────────

def _parse_time(raw) -> int:
    """
    Convert any time value to unix integer seconds.
    Handles:
      19.03.2026 12:00:00.000 UTC   (Dukascopy / TradingView export)
      2026-03-19 12:00:00
      2026-03-19T12:00:00Z
      2026.03.19 12:00
      unix int seconds
      unix int milliseconds (> 1e10)
    """
    if isinstance(raw, (int, float)):
        t = float(raw)
        return int(t / 1000) if t > 1e10 else int(t)

    if not isinstance(raw, str):
        return int(_time.time())

    # Clean common suffixes and milliseconds
    s = raw.strip()
    s = re.sub(r'\s*UTC\s*$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\.\d{3}$', '', s)          # strip .000 ms
    s = s.strip()

    # Try formats in order — most specific first
    FMTS = [
        "%d.%m.%Y %H:%M:%S",    # 19.03.2026 12:00:00  (Dukascopy)
        "%d.%m.%Y %H:%M",       # 19.03.2026 12:00
        "%Y-%m-%dT%H:%M:%S",    # 2026-03-19T12:00:00
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",    # 2026-03-19 12:00:00
        "%Y-%m-%d %H:%M",
        "%Y.%m.%d %H:%M:%S",    # 2026.03.19 12:00:00
        "%Y.%m.%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d",
    ]

    for fmt in FMTS:
        try:
            dt = datetime.datetime.strptime(s[:len(fmt) + 2], fmt)
            return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        except ValueError:
            continue

    # Last resort: try dateutil if available
    try:
        from dateutil import parser as dp
        return int(dp.parse(s).timestamp())
    except Exception:
        pass

    return int(_time.time())


# ── NORMALIZER ────────────────────────────────────────────────────────────────

def _normalize_bars(bars: List[Dict]) -> List[Dict]:
    """
    Ensure every bar has time as plain int unix seconds.
    Deduplicates and sorts ascending.
    """
    out = []
    for b in bars:
        raw_t = (b.get('time') or b.get('timestamp') or b.get('UTC') or
                 b.get('datetime') or b.get('date') or b.get('Datetime') or
                 b.get('Date') or 0)
        t = _parse_time(raw_t)
        try:
            out.append({
                'time':   t,
                'open':   float(b.get('open',   b.get('Open',   b.get('o', 0)))),
                'high':   float(b.get('high',   b.get('High',   b.get('h', 0)))),
                'low':    float(b.get('low',    b.get('Low',    b.get('l', 0)))),
                'close':  float(b.get('close',  b.get('Close',  b.get('c', 0)))),
                'volume': float(b.get('volume', b.get('Volume', b.get('vol', b.get('v', 100))))),
            })
        except (TypeError, ValueError):
            continue

    out.sort(key=lambda x: x['time'])
    seen: set = set()
    deduped = []
    for bar in out:
        if bar['time'] not in seen:
            seen.add(bar['time'])
            deduped.append(bar)
    return deduped


# ── CSV ───────────────────────────────────────────────────────────────────────

def load_csv_text(text: str) -> Optional[List[Dict]]:
    """
    Parse OHLCV CSV. Auto-detects delimiter and column names.
    Handles Dukascopy, TradingView, MT4, and generic formats.
    """
    text = text.strip()
    if not text:
        return None

    # Detect delimiter from first line
    first_line = text.split('\n')[0]
    delim = ','
    for d in [',', '\t', ';']:
        if d in first_line:
            delim = d
            break

    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    if not reader.fieldnames:
        return None

    # Normalize field names for lookup
    fields_lower = {f.strip().lower().replace('"', '').replace("'", ''): f
                    for f in reader.fieldnames}

    def find_col(*candidates):
        for name in candidates:
            if name in fields_lower:
                return fields_lower[name]
            # partial match
            for k, v in fields_lower.items():
                if name in k:
                    return v
        return None

    time_col  = find_col('utc', 'datetime', 'timestamp', 'time', 'date')
    open_col  = find_col('open')
    high_col  = find_col('high')
    low_col   = find_col('low')
    close_col = find_col('close')
    vol_col   = find_col('volume', 'vol', 'tick')

    if not all([open_col, high_col, low_col, close_col]):
        return None

    bars = []
    for row in reader:
        try:
            o = float(row[open_col])
            h = float(row[high_col])
            l = float(row[low_col])
            c = float(row[close_col])
            v = float(row[vol_col]) if vol_col and row.get(vol_col, '').strip() else 100.0
        except (ValueError, TypeError):
            continue

        if any(math.isnan(x) for x in [o, h, l, c]):
            continue

        t_raw = row[time_col].strip() if time_col and row.get(time_col) else ''
        bars.append({
            'time': t_raw,
            'open': o, 'high': h, 'low': l, 'close': c, 'volume': v,
        })

    if len(bars) < 3:
        return None

    return _normalize_bars(bars)


# ── BITGET ────────────────────────────────────────────────────────────────────

BITGET_GRAN = {
    '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min',
    '30m': '30min', '1h': '1H', '4h': '4H', '1d': '1D',
}

async def fetch_bitget(api_key: str, api_secret: str, symbol: str,
                       granularity: str = '5m', limit: int = 300) -> List[Dict]:
    gran = BITGET_GRAN.get(granularity.lower(), granularity)
    url = 'https://api.bitget.com/api/v2/spot/market/candles'
    params = {'symbol': symbol.upper(), 'granularity': gran,
              'limit': str(min(limit, 1000))}
    headers = {'ACCESS-KEY': api_key, 'Content-Type': 'application/json'}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    if data.get('code') != '00000':
        raise ValueError(f"Bitget error: {data.get('msg', 'unknown')}")
    bars = []
    for c in data.get('data', []):
        bars.append({
            'time': int(c[0]),      # ms — normalizer converts
            'open': float(c[1]), 'high': float(c[2]),
            'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5]),
        })
    return _normalize_bars(bars)


# ── OANDA ─────────────────────────────────────────────────────────────────────

OANDA_GRAN = {
    '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
    '1h': 'H1', '4h': 'H4', '1d': 'D',
}

async def fetch_oanda(token: str, account_id: str,
                      instrument: str = 'EUR_USD',
                      granularity: str = 'M5',
                      count: int = 300) -> List[Dict]:
    gran = OANDA_GRAN.get(granularity.lower(), granularity)
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    params = {'count': str(min(count, 5000)), 'granularity': gran, 'price': 'M'}
    for host in ['https://api-fxtrade.oanda.com', 'https://api-fxpractice.oanda.com']:
        url = f'{host}/v3/instruments/{instrument}/candles'
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, params=params, headers=headers)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                bars = []
                for c in data.get('candles', []):
                    if not c.get('complete', True):
                        continue
                    mid = c.get('mid', {})
                    bars.append({
                        'time':   c['time'][:19],
                        'open':   float(mid.get('o', 0)),
                        'high':   float(mid.get('h', 0)),
                        'low':    float(mid.get('l', 0)),
                        'close':  float(mid.get('c', 0)),
                        'volume': float(c.get('volume', 100)),
                    })
                if bars:
                    return _normalize_bars(bars)
        except Exception:
            continue
    raise ValueError('OANDA connection failed — check token')


# ── SAMPLE ────────────────────────────────────────────────────────────────────

def generate_sample(n: int = 300, instrument: str = 'EURUSD',
                    tf_seconds: int = 300) -> List[Dict]:
    bars = []
    p = 1.1050
    t = int(_time.time()) - n * tf_seconds
    for i in range(n):
        phase = (0 if i < int(n * .25) else 1 if i < int(n * .45)
                 else 2 if i < int(n * .75) else 3)
        noise = [0.0008, 0.0015, 0.0025, 0.0012][phase]
        trend = [0.0, 0.0003 * math.sin(i * .3), 0.0012, -0.0005][phase]
        o = p
        c = o + random.gauss(trend, noise)
        wick_h = random.expovariate(1 / 0.0004)
        wick_l = random.expovariate(1 / 0.0004)
        if i == int(n * .44):
            wick_h = 0.0025
        h = max(o, c) + wick_h
        l = min(o, c) - wick_l
        vol = 400 + random.expovariate(1 / 300)
        if phase == 2:
            vol *= 2.2
        if i in [int(n * .25), int(n * .45)]:
            vol *= 3.5
        bars.append({
            'time': t + i * tf_seconds,
            'open': round(o, 5), 'high': round(h, 5),
            'low': round(l, 5), 'close': round(c, 5),
            'volume': int(vol),
        })
        p = c
    return _normalize_bars(bars)
