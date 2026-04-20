"""
data_connectors.py  —  QUIMERIA SMK data sources.
Supports: MT4, MT5, TradingView, Dukascopy, Bitget, OANDA, generic CSV.
Every loader returns List[Dict] with: time(int unix s), open, high, low, close, volume.
"""
import csv, io, math, random, re, time as _time, datetime
from typing import List, Dict, Optional
import httpx


# ── TIME PARSER ───────────────────────────────────────────────────────────────

def _parse_time(raw) -> int:
    if isinstance(raw, (int, float)):
        t = float(raw)
        return int(t / 1000) if t > 1e10 else int(t)
    if not isinstance(raw, str) or not raw.strip():
        return int(_time.time())

    s = raw.strip().replace('"', '').replace("'", '')
    # Strip timezone suffix and milliseconds
    s = re.sub(r'\s*UTC\s*$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'[+-]\d{2}:\d{2}$', '', s)      # strip +00:00
    s = re.sub(r'\.\d+$', '', s)                  # strip .000 ms
    s = s.strip()

    FMTS = [
        "%Y.%m.%d %H:%M:%S",    # MT4/MT5:  2024.01.15 00:00:00
        "%Y.%m.%d %H:%M",       # MT4:      2024.01.15 00:00
        "%d.%m.%Y %H:%M:%S",    # Dukascopy: 19.03.2026 12:00:00
        "%d.%m.%Y %H:%M",
        "%Y-%m-%dT%H:%M:%S",    # TradingView ISO
        "%Y-%m-%d %H:%M:%S",    # generic
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S",    # TradingView US
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d",
    ]
    for fmt in FMTS:
        try:
            dt = datetime.datetime.strptime(s[:len(fmt)+2].strip(), fmt)
            return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        except ValueError:
            continue

    try:
        from dateutil import parser as dp
        return int(dp.parse(s).timestamp())
    except Exception:
        pass

    return int(_time.time())


def _parse_time_pair(date_str: str, time_str: str) -> int:
    """Parse separate date + time columns (MT4, MT5, TradingView US format)."""
    combined = (date_str.strip() + ' ' + time_str.strip()).strip()
    return _parse_time(combined)


# ── NORMALIZER ────────────────────────────────────────────────────────────────

def _normalize_bars(bars: List[Dict]) -> List[Dict]:
    out = []
    for b in bars:
        t = (b.get('time') or b.get('timestamp') or b.get('UTC') or
             b.get('datetime') or b.get('date') or b.get('Datetime') or
             b.get('Date') or 0)
        parsed_t = _parse_time(t)
        try:
            out.append({
                'time':   parsed_t,
                'open':   float(b.get('open',   b.get('Open',   b.get('o', 0)))),
                'high':   float(b.get('high',   b.get('High',   b.get('h', 0)))),
                'low':    float(b.get('low',    b.get('Low',    b.get('l', 0)))),
                'close':  float(b.get('close',  b.get('Close',  b.get('c', 0)))),
                'volume': float(b.get('volume', b.get('Volume', b.get('tickvol',
                          b.get('TickVol', b.get('vol', b.get('VOL',
                          b.get('v', 100)))))))),
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


# ── FORMAT DETECTOR ───────────────────────────────────────────────────────────

def _detect_format(header_line: str, first_data_line: str) -> str:
    h = header_line.lower().replace('<', '').replace('>', '').strip()
    if 'date' in h and 'time' in h and ('vol' in h or 'tickvol' in h):
        if '.' in first_data_line.split(',')[0]:
            return 'MT4_MT5'        # 2024.01.15, 00:00
        return 'TV_US'              # 01/15/2024, 00:00
    if 'time' in h and 'open' in h and 'T' in first_data_line:
        return 'TV_ISO'             # 2024-01-15T00:00:00+00:00
    if 'utc' in h:
        return 'DUKASCOPY'          # 19.03.2026 12:00:00.000 UTC
    return 'GENERIC'


# ── CSV PARSERS ───────────────────────────────────────────────────────────────

def _parse_mt4_mt5(text: str, delim: str) -> Optional[List[Dict]]:
    """
    MT4:  <DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>
          2024.01.15,00:00,1.105,1.106,1.104,1.105,1245
    MT5:  DATE,TIME,OPEN,HIGH,LOW,CLOSE,TICKVOL,VOL,SPREAD
          2024.01.15,00:00:00,1.105,1.106,1.104,1.105,1245,0,2
    """
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    rows = list(reader)
    if len(rows) < 2:
        return None

    # Clean header — strip < >
    hdr = [h.strip().lower().replace('<','').replace('>','') for h in rows[0]]
    di = hdr.index('date') if 'date' in hdr else -1
    ti = hdr.index('time') if 'time' in hdr else -1
    oi = next((i for i,h in enumerate(hdr) if h == 'open'), -1)
    hi = next((i for i,h in enumerate(hdr) if h == 'high'), -1)
    li = next((i for i,h in enumerate(hdr) if h == 'low'), -1)
    ci = next((i for i,h in enumerate(hdr) if h == 'close'), -1)
    vi = next((i for i,h in enumerate(hdr) if h in ('vol','tickvol','volume')), -1)

    if -1 in (di, ti, oi, hi, li, ci):
        return None

    bars = []
    for row in rows[1:]:
        if len(row) <= ci:
            continue
        try:
            t = _parse_time_pair(row[di], row[ti])
            o = float(row[oi]); hh = float(row[hi])
            ll = float(row[li]); c = float(row[ci])
            v = float(row[vi]) if vi >= 0 and vi < len(row) else 100.0
            bars.append({'time': t, 'open': o, 'high': hh, 'low': ll,
                         'close': c, 'volume': v})
        except (ValueError, IndexError):
            continue
    return bars if len(bars) >= 3 else None


def _parse_tv_iso(text: str, delim: str) -> Optional[List[Dict]]:
    """
    TradingView ISO export:
    time,open,high,low,close,Volume
    2024-01-15T00:00:00+00:00,1.105,1.106,1.104,1.105,1245
    """
    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    fields = {f.strip().lower(): f for f in (reader.fieldnames or [])}

    def fc(*names):
        for n in names:
            if n in fields: return fields[n]
        return None

    time_col  = fc('time', 'datetime', 'date')
    open_col  = fc('open')
    high_col  = fc('high')
    low_col   = fc('low')
    close_col = fc('close')
    vol_col   = fc('volume', 'vol')

    if not all([time_col, open_col, high_col, low_col, close_col]):
        return None

    bars = []
    for row in reader:
        try:
            t = _parse_time(row[time_col])
            bars.append({
                'time':   t,
                'open':   float(row[open_col]),
                'high':   float(row[high_col]),
                'low':    float(row[low_col]),
                'close':  float(row[close_col]),
                'volume': float(row[vol_col]) if vol_col and row.get(vol_col, '').strip() else 100.0,
            })
        except (ValueError, TypeError):
            continue
    return bars if len(bars) >= 3 else None


def _parse_tv_us(text: str, delim: str) -> Optional[List[Dict]]:
    """
    TradingView US date format:
    Date,Time,Open,High,Low,Close,Volume
    01/15/2024,00:00,1.105,1.106,1.104,1.105,1245
    """
    return _parse_mt4_mt5(text, delim)  # same two-column structure


def _parse_generic(text: str, delim: str) -> Optional[List[Dict]]:
    """Generic OHLCV — single datetime column or any recognizable format."""
    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    if not reader.fieldnames:
        return None

    fields_lower = {f.strip().lower().replace('"','').replace("'",'').
                    replace('<','').replace('>',''): f
                    for f in reader.fieldnames}

    def find(*names):
        for n in names:
            if n in fields_lower: return fields_lower[n]
            for k, v in fields_lower.items():
                if n in k: return v
        return None

    time_col  = find('utc','datetime','timestamp','time','date')
    open_col  = find('open')
    high_col  = find('high')
    low_col   = find('low')
    close_col = find('close')
    vol_col   = find('volume','vol','tickvol','tick')

    if not all([open_col, high_col, low_col, close_col]):
        return None

    bars = []
    for row in reader:
        try:
            t_raw = row[time_col].strip() if time_col and row.get(time_col) else ''
            t = _parse_time(t_raw)
            o = float(row[open_col]); h = float(row[high_col])
            l = float(row[low_col]);  c = float(row[close_col])
            v = float(row[vol_col]) if vol_col and row.get(vol_col,'').strip() else 100.0
            if any(math.isnan(x) for x in [o,h,l,c]): continue
            bars.append({'time':t,'open':o,'high':h,'low':l,'close':c,'volume':v})
        except (ValueError, TypeError):
            continue
    return bars if len(bars) >= 3 else None


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

def load_csv_text(text: str, source_hint: str = 'auto') -> Optional[List[Dict]]:
    """
    Parse OHLCV CSV from any source.
    source_hint: 'auto' | 'mt4' | 'mt5' | 'tradingview' | 'dukascopy' | 'generic'
    """
    text = text.strip()
    if not text:
        return None

    lines = text.split('\n')
    if len(lines) < 2:
        return None

    header = lines[0]
    first_data = lines[1] if len(lines) > 1 else ''

    # Detect delimiter
    delim = ','
    for d in [',', '\t', ';']:
        if d in header:
            delim = d
            break

    # Auto-detect format or use hint
    fmt = source_hint.lower()
    if fmt == 'auto':
        fmt = _detect_format(header, first_data)

    bars = None

    if fmt in ('mt4', 'mt5', 'MT4_MT5'):
        bars = _parse_mt4_mt5(text, delim)
    elif fmt in ('tradingview', 'TV_ISO') or 'T' in first_data and '+' in first_data:
        bars = _parse_tv_iso(text, delim)
        if not bars:
            bars = _parse_mt4_mt5(text, delim)
    elif fmt == 'TV_US':
        bars = _parse_tv_us(text, delim)
    elif fmt in ('dukascopy', 'DUKASCOPY'):
        bars = _parse_generic(text, delim)
    else:
        # Try each parser in order until one works
        for parser in [_parse_mt4_mt5, _parse_tv_iso, _parse_generic]:
            bars = parser(text, delim)
            if bars:
                break

    if not bars:
        return None

    result = _normalize_bars(bars)
    return result if len(result) >= 3 else None


# ── BITGET ────────────────────────────────────────────────────────────────────

BITGET_GRAN = {
    '1m':'1min','3m':'3min','5m':'5min','15m':'15min',
    '30m':'30min','1h':'1H','4h':'4H','1d':'1D',
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
        raise ValueError(f"Bitget error: {data.get('msg','unknown')}")
    bars = []
    for c in data.get('data', []):
        bars.append({'time': int(c[0]), 'open': float(c[1]), 'high': float(c[2]),
                     'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5])})
    return _normalize_bars(bars)


# ── OANDA ─────────────────────────────────────────────────────────────────────

OANDA_GRAN = {
    '1m':'M1','5m':'M5','15m':'M15','30m':'M30',
    '1h':'H1','4h':'H4','1d':'D',
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
                if resp.status_code != 200: continue
                data = resp.json()
                bars = []
                for c in data.get('candles', []):
                    if not c.get('complete', True): continue
                    mid = c.get('mid', {})
                    bars.append({'time': c['time'][:19], 'open': float(mid.get('o',0)),
                                 'high': float(mid.get('h',0)), 'low': float(mid.get('l',0)),
                                 'close': float(mid.get('c',0)), 'volume': float(c.get('volume',100))})
                if bars:
                    return _normalize_bars(bars)
        except Exception:
            continue
    raise ValueError('OANDA connection failed')


# ── SAMPLE ────────────────────────────────────────────────────────────────────

def generate_sample(n: int = 300, instrument: str = 'EURUSD',
                    tf_seconds: int = 300) -> List[Dict]:
    bars = []
    p = 1.1050
    t = int(_time.time()) - n * tf_seconds
    for i in range(n):
        phase = (0 if i < int(n*.25) else 1 if i < int(n*.45)
                 else 2 if i < int(n*.75) else 3)
        noise = [0.0008, 0.0015, 0.0025, 0.0012][phase]
        trend = [0.0, 0.0003*math.sin(i*.3), 0.0012, -0.0005][phase]
        o = p
        c = o + random.gauss(trend, noise)
        wh = random.expovariate(1/0.0004)
        wl = random.expovariate(1/0.0004)
        if i == int(n*.44): wh = 0.0025
        h = max(o,c)+wh; l = min(o,c)-wl
        vol = 400 + random.expovariate(1/300)
        if phase == 2: vol *= 2.2
        if i in [int(n*.25), int(n*.45)]: vol *= 3.5
        bars.append({'time': t+i*tf_seconds, 'open': round(o,5),
                     'high': round(h,5), 'low': round(l,5),
                     'close': round(c,5), 'volume': int(vol)})
        p = c
    return _normalize_bars(bars)
