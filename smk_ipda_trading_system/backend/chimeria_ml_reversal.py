"""
Port of SMK's IPDA Reversal Prediction ML model
Uses XGBoost to predict high-probability reversal windows
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import pickle
import os
from dataclasses import dataclass

@dataclass
class ReversalPrediction:
    probability: float
    is_reversal: bool
    confidence: str
    features: Dict[str, float]

class MLReversalPredictor:
    """XGBoost-based IPDA reversal predictor"""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.threshold = 0.35
        if model_path and os.path.exists(model_path):
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier()
                self.model.load_model(model_path)
                print(f"[ML] Loaded reversal model from {model_path}")
            except Exception as e:
                print(f"[ML] Failed to load model: {e}")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer IPDA features for reversal prediction"""
        f = df.copy()
        close = f['close']
        high = f['high']
        low = f['low']

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        f['atr_14'] = tr.rolling(14).mean()
        f['atr_pct'] = f['atr_14'] / (close + 1e-9)

        # IPDA 20/40/60 ranges
        for w in [20, 40, 60]:
            f[f'ipda_{w}d_high'] = high.rolling(w).max()
            f[f'ipda_{w}d_low'] = low.rolling(w).min()
            range_val = f[f'ipda_{w}d_high'] - f[f'ipda_{w}d_low']
            f[f'ipda_{w}d_pos'] = (close - f[f'ipda_{w}d_low']) / (range_val + 1e-9)
            f[f'breach_high_{w}d'] = (high >= f[f'ipda_{w}d_high']).astype(int)
            f[f'breach_low_{w}d'] = (low <= f[f'ipda_{w}d_low']).astype(int)

            eq = f[f'ipda_{w}d_low'] + range_val * 0.5
            f[f'above_equil_{w}d'] = (close > eq).astype(int)

        # FVG detection
        f['bull_fvg'] = ((low > high.shift(2))).astype(int)
        f['bear_fvg'] = ((high < low.shift(2))).astype(int)

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        f['rsi_14'] = 100 - (100 / (1 + rs))

        # Momentum
        f['momentum_5'] = close.pct_change(5)
        f['momentum_10'] = close.pct_change(10)
        f['momentum_20'] = close.pct_change(20)

        return f

    def predict(self, df: pd.DataFrame) -> ReversalPrediction:
        """Predict reversal probability from OHLCV data"""
        if self.model is None:
            return ReversalPrediction(
                probability=0.0,
                is_reversal=False,
                confidence="NO_MODEL",
                features={}
            )

        try:
            features = self.engineer_features(df)

            # Select feature columns (matching training)
            feature_cols = [c for c in features.columns if any(
                x in c for x in ['ipda_', 'breach_', 'above_', 'rsi', 'momentum', 'atr_pct']
            )]

            latest = features[feature_cols].iloc[[-1]].fillna(0)
            prob = float(self.model.predict_proba(latest.values)[0][1])

            is_reversal = prob >= self.threshold
            confidence = "HIGH" if prob > 0.55 else "MEDIUM" if prob > 0.35 else "LOW"

            # Extract key features for display
            key_features = {
                'ipda_60d_pos': float(features['ipda_60d_pos'].iloc[-1]) if 'ipda_60d_pos' in features else 0,
                'atr_pct': float(features['atr_pct'].iloc[-1]) if 'atr_pct' in features else 0,
                'rsi_14': float(features['rsi_14'].iloc[-1]) if 'rsi_14' in features else 0,
            }

            return ReversalPrediction(
                probability=round(prob, 3),
                is_reversal=is_reversal,
                confidence=confidence,
                features=key_features
            )

        except Exception as e:
            print(f"[ML] Prediction error: {e}")
            return ReversalPrediction(
                probability=0.0,
                is_reversal=False,
                confidence=f"ERROR:{e}",
                features={}
            )
