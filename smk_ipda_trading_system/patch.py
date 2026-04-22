#!/usr/bin/env python3
"""
patch.py  —  QUIMERIA universal patcher
Place this at the project root and leave it there forever.

Usage:
    python patch.py

Add patches to the PATCHES list at the bottom of this file.
Each patch is a dict with:
    file  : path relative to this script's directory
    old   : exact string to find (triple-quoted for multiline)
    new   : replacement string
    desc  : short description shown in output
"""
import os

ROOT   = os.path.dirname(os.path.abspath(__file__))
PASSED = []
SKIPPED= []
FAILED = []


def apply(file: str, old: str, new: str, desc: str):
    path = os.path.join(ROOT, file)
    if not os.path.exists(path):
        FAILED.append(f"NOT FOUND    {file}  ({desc})")
        return
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if old not in src:
        if new in src:
            SKIPPED.append(f"ALREADY DONE {file}  ({desc})")
        else:
            FAILED.append(f"NO MATCH     {file}  ({desc})")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(old, new, 1))
    PASSED.append(f"PATCHED      {file}  ({desc})")


# ─────────────────────────────────────────────────────────────────────────────
# PATCHES — add new ones here, old ones stay for reference
# ─────────────────────────────────────────────────────────────────────────────

PATCHES = [

    # ── plugin_manager.py: sys.path guard ────────────────────────────────────
    dict(
        file="backend/plugin_manager.py",
        desc="add sys.path guard so plugins package is always importable",
        old="""\
import importlib
import os
import traceback
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from plugins import SMKPlugin""",
        new="""\
import importlib
import os
import sys
import traceback
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from plugins import SMKPlugin""",
    ),

    # ── equilibrium_cross_detector.py: [7-9] → [20,40,60] ───────────────────
    dict(
        file="core/equilibrium_cross_detector.py",
        desc="lookbacks=[7-9] evaluates to [-2], fix to [20,40,60]",
        old="def __init__(self, lookbacks=[7-9]):",
        new="def __init__(self, lookbacks=[20, 40, 60]):",
    ),

    # ── premium_discount_detector.py: [7-9] bug ──────────────────────────────
    dict(
        file="core/premium_discount_detector.py",
        desc="lookbacks=[7-9] evaluates to [-2], fix to [20,40,60]",
        old="def __init__(self, lookbacks=[7-9]):",
        new="def __init__(self, lookbacks=[20, 40, 60]):",
    ),

    # ── premium_discount_detector.py: ranges[9] KeyError ─────────────────────
    dict(
        file="core/premium_discount_detector.py",
        desc="ranges[9] KeyError — key is max(lookbacks)=60 not 9",
        old="""\
        h60 = ranges[9]['high']
        l60 = ranges[9]['low']""",
        new="""\
        h60 = ranges[max(self.lookbacks)]['high']
        l60 = ranges[max(self.lookbacks)]['low']""",
    ),

    # ── manipulation_detector.py: [11-13] bug ────────────────────────────────
    dict(
        file="lambda_sensors/manipulation_detector.py",
        desc="[11-13] evaluates to [-2], Judas swing never fires",
        old="for lb in [11-13]:",
        new="for lb in [20, 40, 60]:",
    ),

    # ── topological_fracture_detector.py: dgms[14] IndexError ────────────────
    dict(
        file="detectors/topological_fracture_detector.py",
        desc="dgms[14] IndexError — ripser returns 2 elements, H1 is dgms[1]",
        old="h1_intervals = dgms[14] # H1 lifetimes represent structural conflict [15]",
        new="h1_intervals = dgms[1]  # H1 = 1-dimensional loops (dgms[0]=H0, dgms[1]=H1)",
    ),

]
dict(
    file="backend/smk_pipeline.py",
    desc="ensure backend/ on sys.path before importing plugin_manager",
    old="""\
# Plugin system — lazy load to avoid circular imports
_plugin_mgr = None
def _get_plugins():
    global _plugin_mgr
    if _plugin_mgr is None:
        try:
            from plugin_manager import get_plugin_manager
            _plugin_mgr = get_plugin_manager()
        except Exception as e:
            print(f"[SMK] Plugin manager not available: {e}")
            _plugin_mgr = None
    return _plugin_mgr""",
    new="""\
# Plugin system — lazy load to avoid circular imports
_plugin_mgr = None
_plugin_mgr_failed = False
def _get_plugins():
    global _plugin_mgr, _plugin_mgr_failed
    if _plugin_mgr_failed:
        return None
    if _plugin_mgr is None:
        try:
            import sys as _sys
            _bd = os.path.dirname(os.path.abspath(__file__))
            if _bd not in _sys.path:
                _sys.path.insert(0, _bd)
            from plugin_manager import get_plugin_manager
            _plugin_mgr = get_plugin_manager()
        except Exception as e:
            print(f"[SMK] Plugin manager not available: {e}")
            _plugin_mgr_failed = True
            _plugin_mgr = None
    return _plugin_mgr""",
),
# ─────────────────────────────────────────────────────────────────────────────

for p in PATCHES:
    apply(**p)

print()
print("=" * 60)
print("  QUIMERIA PATCH RESULTS")
print("=" * 60)
for m in PASSED:   print(f"  ✓  {m}")
for m in SKIPPED:  print(f"  -  {m}")
for m in FAILED:   print(f"  ✗  {m}")
print()
print(f"  {len(PASSED)} applied   {len(SKIPPED)} skipped   {len(FAILED)} failed")
print("=" * 60)
if PASSED:
    print()
    print("  Restart uvicorn to pick up changes.")
print()
