"""
jax_shim.py — drop-in replacement when JAX is not installed.
Place this in the backend/ folder. SMK modules that import jax will get
numpy-backed equivalents with zero code changes needed.
"""
import sys, types, numpy as np

# Only install shim if real JAX is missing
try:
    import jax as _real_jax
    import jax.numpy as _real_jnp
    print("[jax_shim] Real JAX found — shim not needed")
except ImportError:
    print("[jax_shim] JAX not installed — installing numpy shim")

    # ── jax.numpy shim ────────────────────────────────────────────────────────
    jnp_mod = types.ModuleType("jax.numpy")
    # Delegate everything to numpy
    for _attr in dir(np):
        try:
            setattr(jnp_mod, _attr, getattr(np, _attr))
        except Exception:
            pass
    # extras jax.numpy has that numpy spells differently
    jnp_mod.DeviceArray = np.ndarray
    jnp_mod.ndarray = np.ndarray

    # ── jax shim ──────────────────────────────────────────────────────────────
    jax_mod = types.ModuleType("jax")
    jax_mod.numpy = jnp_mod

    def _jit(fn, *a, **kw):
        """No-op JIT — just return the function unchanged."""
        return fn

    def _grad(fn, *a, **kw):
        raise NotImplementedError("jax.grad not available in shim")

    jax_mod.jit    = _jit
    jax_mod.grad   = _grad
    jax_mod.vmap   = lambda fn, *a, **kw: fn
    jax_mod.lax    = types.SimpleNamespace(scan=None, cond=None)

    # Register both jax and jax.numpy in sys.modules
    sys.modules["jax"]         = jax_mod
    sys.modules["jax.numpy"]   = jnp_mod
    sys.modules["jaxlib"]      = types.ModuleType("jaxlib")

    print("[jax_shim] Shim installed: jax + jax.numpy → numpy backend")
