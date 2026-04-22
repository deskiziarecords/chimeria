"""
plugins/__init__.py  —  QUIMERIA SMK Plugin System
Each plugin is a self-contained detector that receives a bar + window DataFrame
and returns a flat telemetry dict that gets merged into the SMK bar payload.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np


class SMKPlugin(ABC):
    """
    Base class for all QUIMERIA SMK plugins.

    Implement update() — return a flat dict of serializable values.
    All values must be plain Python types (bool, int, float, str, list).
    Never return numpy scalars — use float(), int(), bool() explicitly.
    """
    name:            str   = "unnamed"
    layer:           str   = "L3-ext"     # L1 L2 L3 L4 λ-ext EXE
    sensor_id:       str   = "s99"        # shown in frontend sensor bar
    enabled:         bool  = True
    requires_warmup: int   = 20           # minimum bars before firing

    def __init__(self):
        self._bar_count = 0
        self._ready     = False

    def on_bar(self, bar: dict, df: pd.DataFrame, smk: dict) -> dict:
        """Called by plugin_manager on every bar. Handles warmup."""
        self._bar_count += 1
        if self._bar_count < self.requires_warmup:
            return self._warmup_payload()
        self._ready = True
        try:
            return self.update(bar, df, smk)
        except Exception as e:
            return {"status": f"ERROR:{e}", "active": False, "score": 0.0}

    @abstractmethod
    def update(self, bar: dict, df: pd.DataFrame, smk: dict) -> dict:
        """
        Process one bar.
        bar  — current OHLCV dict {time, open, high, low, close, volume}
        df   — rolling window DataFrame (up to plugin's lookback)
        smk  — current SMK result dict (read-only, partial — earlier layers done)
        Returns flat dict — all values must be JSON-serializable plain Python types.
        """
        ...

    def _warmup_payload(self) -> dict:
        return {"status": "WARMUP", "active": False, "score": 0.0}

    def reset(self):
        self._bar_count = 0
        self._ready     = False
