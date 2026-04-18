"""
QUIMERIA SMK Backend Package

This package contains the FastAPI backend for the Sovereign Market Kernel (SMK).
"""

__version__ = "1.0.0"
__author__ = "QUIMERIA / SMK Team"

# Make key components easily importable from the backend package

from .main import app as fastapi_app
from .smk_pipeline import SMKPipeline

# Optional: expose data connectors for convenience
try:
    from .data_connectors import (
        load_csv_text,
        fetch_bitget,
        fetch_oanda,
        generate_sample
    )
except ImportError:
    # Graceful fallback if files are not yet created
    pass

__all__ = [
    "fastapi_app",
    "SMKPipeline",
    "load_csv_text",
    "fetch_bitget",
    "fetch_oanda",
    "generate_sample",
]
