# ipda_utils.py
"""
IPDA Feature Engineering Utilities for Reversal Prediction
Based on ICT/IPDA concepts: 20/40/60 ranges, equilibrium, premium/discount, liquidity sweeps, etc.
"""

import numpy as np
import pandas as pd

def engineer_ipda_features(df: pd.DataFrame, windows: list = [20, 40, 60]) -> pd.DataFrame:
    """
    Engineer IPDA-inspired features from OHLCV DataFrame.
    Assumes df has columns: open, high, low, close, volume (index = datetime)
    """
    df = df.copy()
    
    # Basic technicals
    df['returns'] = df['close'].pct_change()
    df['atr_14'] = _calculate_atr(df, 14)
    
    for w in windows:
        prefix = f"ipda_{w}d"
        
        # Rolling High / Low (IPDA structural levels)
        df[f"{prefix}_high"] = df['high'].rolling(window=w).max()
        df[f"{prefix}_low"] = df['low'].rolling(window=w).min()
        
        # Equilibrium (Fair Value / 50% level) - mean of midpoints
        df[f"{prefix}_eq"] = (df[f"{prefix}_high"] + df[f"{prefix}_low"]) / 2
        
        # Premium / Discount Zone
        df[f"{prefix}_zone"] = np.where(
            df['close'] > df[f"{prefix}_eq"], 1,  # Premium (short bias)
            np.where(df['close'] < df[f"{prefix}_eq"], -1, 0)  # Discount (long bias)
        )
        
        # Distance to Equilibrium (in ATR units)
        df[f"{prefix}_dist_to_eq"] = (df['close'] - df[f"{prefix}_eq"]) / (df['atr_14'] + 1e-9)
        
        # Range Width & Coherence (how aligned the ranges are)
        df[f"{prefix}_range"] = df[f"{prefix}_high"] - df[f"{prefix}_low"]
        # Simple coherence proxy: current range vs average of previous windows
        if w == windows[-1]:  # for largest window
            df[f"{prefix}_coherence"] = df[f"{prefix}_range"] / (df[f"ipda_{windows[0]}d_range"].rolling(10).mean() + 1e-9)
    
    # Displacement / Expansion signals (large candle relative to ATR)
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = df['body'] / (df['range'] + 1e-9)
    df['displacement'] = ((df['range'] > 1.5 * df['atr_14']) & (df['body_ratio'] > 0.7)).astype(int)
    
    # Volatility Decay / Entrapment (low volatility → potential reversal setup)
    df['vol_ratio'] = df['range'].rolling(20).mean() / (df['atr_14'] + 1e-9)
    df['entrapped'] = (df['vol_ratio'] < 0.75).astype(int)
    
    # Liquidity Sweep Proxy (price breaching recent high/low then reversing)
    for w in windows:
        df[f"ipda_{w}d_breach_high"] = (df['high'] > df[f"ipda_{w}d_high"].shift(1)).astype(int)
        df[f"ipda_{w}d_breach_low"] = (df['low'] < df[f"ipda_{w}d_low"].shift(1)).astype(int)
    
    # Fair Value Gap style imbalance (simple 3-candle gap)
    df['fvg_bull'] = (df['low'] > df['high'].shift(2)).astype(int)
    df['fvg_bear'] = (df['high'] < df['low'].shift(2)).astype(int)
    
    # Session / Time features (basic)
    df['dayofweek'] = df.index.dayofweek
    df['hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
    
    # Momentum & Mean Reversion
    df['rsi_14'] = _calculate_rsi(df['close'], 14)
    df['mom_10'] = df['close'].pct_change(10)
    
    print(f"   → Engineered {len([c for c in df.columns if 'ipda' in c or 'displacement' in c or 'entrapped' in c])} IPDA-related features")
    return df


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Simple ATR calculation"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Simple RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))