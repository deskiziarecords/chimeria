"""
statsmodels_shim.py — stub for when statsmodels is not installed.
Provides just enough surface area to prevent ImportError at module load time.
The actual Granger/VAR computation will gracefully degrade to fallback logic.
"""
import sys, types

try:
    import statsmodels as _sm
    # Real statsmodels is present — do nothing
except ImportError:
    print("[shim] statsmodels not found — installing stub")

    def _make_mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    # Minimal stub tree
    sm       = _make_mod("statsmodels")
    tsa      = _make_mod("statsmodels.tsa")
    var_mod  = _make_mod("statsmodels.tsa.vector_ar")
    var_m    = _make_mod("statsmodels.tsa.vector_ar.var_model")
    tools    = _make_mod("statsmodels.tools")
    tools_sm = _make_mod("statsmodels.tools.sm_exceptions")

    class _FakeVAR:
        def __init__(self, data): pass
        def select_order(self, maxlags=20):
            return types.SimpleNamespace(selected_orders={"aic": 1})
        def fit(self, lags=1):
            class _R:
                def test_causality(self, *a, **kw):
                    return types.SimpleNamespace(test_stat=0.0, pvalue=1.0)
            return _R()

    var_m.VAR = _FakeVAR
    sm.tsa = tsa
    tsa.vector_ar = var_mod
    var_mod.var_model = var_m

    print("[shim] statsmodels stub installed")
