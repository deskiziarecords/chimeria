"""
backend/data_connectors.py
Data loading utilities for the SMK FastAPI backend.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta
import asyncio

async def load_csv_text(csv_text: str) -> List[Dict]:
    """Parse CSV text into list of bar dicts"""
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(csv_text))
        
        # Standardize columns
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Try to detect datetime column
        datetime_col = None
        for col in ['datetime', 'timestamp', 'time', 'date']:
            if col in df.columns:
                datetime_col = col
                break
        
        if datetime_col:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df.set_index(datetime_col, inplace=True)
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                df[col] = 0.0 if col != 'volume' else 0
        
        # Convert to list of dicts for JSON
        bars = []
        for idx, row in df.iterrows():
            bar = {
                "timestamp": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            }
            bars.append(bar)
        
        return bars
    except Exception as e:
        print(f"CSV parsing error: {e}")
        return []


async def fetch_bitget(api_key: str, api_secret: str, symbol: str, granularity: str, limit: int) -> List[Dict]:
    """Placeholder - replace with real ccxt implementation"""
    print(f"Fetching {limit} bars of {symbol} from Bitget...")
    # Dummy data for development
    await asyncio.sleep(0.5)
    return generate_sample(limit, symbol)


async def fetch_oanda(token: str, account_id: str, instrument: str, granularity: str, count: int) -> List[Dict]:
    """Placeholder - replace with real Oanda API"""
    print(f"Fetching {count} bars of {instrument} from Oanda...")
    await asyncio.sleep(0.5)
    return generate_sample(count, instrument)


def generate_sample(count: int = 300, symbol: str = "EURUSD") -> List[Dict]:
    """Generate realistic sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=count, freq='5min')
    
    # Random walk with some trend
    close = 1.0850 + np.cumsum(np.random.randn(count) * 0.0003)
    high = close + np.abs(np.random.randn(count) * 0.0004)
    low = close - np.abs(np.random.randn(count) * 0.0004)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    
    volume = np.random.randint(1000, 15000, count).astype(float)
    
    bars = []
    for i in range(count):
        bars.append({
            "timestamp": dates[i].isoformat(),
            "open": float(open_[i]),
            "high": float(high[i]),
            "low": float(low[i]),
            "close": float(close[i]),
            "volume": float(volume[i])
        })
    return bars
