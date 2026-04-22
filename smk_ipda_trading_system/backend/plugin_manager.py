"""
plugin_manager.py  —  QUIMERIA SMK Plugin Manager
Discovers all plugins in backend/plugins/, runs them on every bar,
merges telemetry into the SMK payload, exposes REST endpoints.
"""
from __future__ import annotations

import importlib
import os
import traceback
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from plugins import SMKPlugin


# ─────────────────────────────────────────────────────────────────────────────
# PLUGIN REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

# All available plugins — (module_path, class_name)
PLUGIN_REGISTRY = [
    ("plugins.market_rhythm",      "MarketRhythmPlugin"),
    ("plugins.market_heuristic",   "MarketHeuristicPlugin"),
    ("plugins.market_vision",      "MarketVisionPlugin"),
    ("plugins.market_seismology",  "MarketSeismologyPlugin"),
    ("plugins.kali_forensics",     "FileCarvingPlugin"),
    ("plugins.kali_forensics",     "SignatureScanPlugin"),
]


class PluginManager:
    """
    Loads, manages, and runs all SMK plugins.
    Integrates cleanly with SMKPipeline.step() output.

    Usage in smk_pipeline.py:
        from plugin_manager import PluginManager
        self.plugin_mgr = PluginManager()
        self.plugin_mgr.load_all()

        # At end of step():
        r["plugins"] = self.plugin_mgr.run(bar, df, r)
    """

    def __init__(self):
        self.plugins:  List[SMKPlugin] = []
        self.disabled: set[str]        = set()
        self._errors:  Dict[str, str]  = {}

    def load_all(self) -> None:
        """Import and instantiate all registered plugins."""
        loaded = 0
        for module_path, class_name in PLUGIN_REGISTRY:
            try:
                mod    = importlib.import_module(module_path)
                cls    = getattr(mod, class_name)
                plugin = cls()
                self.plugins.append(plugin)
                loaded += 1
                print(f"[PLUGIN] Loaded: {plugin.layer:8s} {plugin.name}")
            except Exception as e:
                self._errors[class_name] = str(e)
                print(f"[PLUGIN] SKIP {class_name}: {e}")

        print(f"[PLUGIN] {loaded}/{len(PLUGIN_REGISTRY)} plugins loaded")

    def disable(self, name: str) -> None:
        self.disabled.add(name)

    def enable(self, name: str) -> None:
        self.disabled.discard(name)

    def set_enabled(self, enabled_names: List[str]) -> None:
        """Enable only the listed plugins by name."""
        all_names = {p.name for p in self.plugins}
        self.disabled = all_names - set(enabled_names)

    def get_status(self) -> List[dict]:
        return [
            {
                "name":    p.name,
                "layer":   p.layer,
                "sensor":  p.sensor_id,
                "enabled": p.name not in self.disabled,
                "ready":   p._ready,
                "warmup":  p.requires_warmup,
                "bars":    p._bar_count,
            }
            for p in self.plugins
        ]

    def run(self, bar: dict, df: pd.DataFrame, smk: dict) -> dict:
        """
        Run all enabled plugins on the current bar.
        Returns a dict keyed by plugin.name with each plugin's telemetry.
        Also appends to the existing smk["sensors"] list if present.
        """
        results: dict = {}

        for plugin in self.plugins:
            if not plugin.enabled or plugin.name in self.disabled:
                continue
            try:
                result = plugin.on_bar(bar, df, smk)
                results[plugin.name] = result
            except Exception as e:
                results[plugin.name] = {
                    "status": f"RUNTIME_ERROR: {e}",
                    "active": False,
                    "score":  0.0,
                }
                print(f"[PLUGIN] Runtime error in {plugin.name}: {e}")

        return results

    def reset(self) -> None:
        """Reset all plugin warmup counters (called on data reload)."""
        for plugin in self.plugins:
            plugin.reset()

    def to_sensor_rows(self, plugin_results: dict) -> List[dict]:
        """
        Convert plugin results to sensor-row format for the frontend.
        Appends after the existing 14 SMK sensors.
        """
        rows = []
        for plugin in self.plugins:
            if plugin.name not in plugin_results:
                continue
            r = plugin_results[plugin.name]
            rows.append({
                "id":     str(plugin.sensor_id),
                "name":   str(plugin.name[:14]),
                "score":  float(r.get("score_norm", r.get("score", 0.0))),
                "active": bool(r.get("active", False)),
                "layer":  str(plugin.layer),
                "status": str(r.get("status", "--")),
            })
        return rows

    @property
    def load_errors(self) -> Dict[str, str]:
        return self._errors


# Global singleton
_plugin_mgr: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    global _plugin_mgr
    if _plugin_mgr is None:
        _plugin_mgr = PluginManager()
        _plugin_mgr.load_all()
    return _plugin_mgr
