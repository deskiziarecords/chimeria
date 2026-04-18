"""
smk_pipeline.py

SMK Pipeline - Orchestrates the full Sovereign Market Kernel
for the FastAPI backend. Handles bar loading, stepping, and status reporting.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime

# Import your core SMK components
from SovereignMarketKernel import SovereignMarketKernel   # We'll create this next if needed
# from core.ipda_phase_detector import IPDACompiler
# ... you can import directly if you prefer flat imports

class SMKPipeline:
    """
    High-level pipeline that wraps the SovereignMarketKernel
    for easy use in FastAPI (loading bars, stepping through them, etc.)
    """

    def __init__(self):
        self.kernel = SovereignMarketKernel()   # Your main orchestrator
        self.raw_bars: List[Dict] = []          # Raw OHLCV data (list of dicts for JSON)
        self.df: Optional[pd.DataFrame] = None  # Processed DataFrame for the kernel
        self.cursor = 0                         # Current bar index for streaming
        self.running = False
        self.last_result: Optional[Dict] = None

        print("✅ SMK Pipeline initialized")

    def load_bars(self, bars: List[Dict]):
        """Load new bars from CSV, Bitget, Oanda, etc."""
        if not bars:
            return

        self.raw_bars = bars
        self.df = self._convert_to_dataframe(bars)
        self.cursor = 0
        self.last_result = None

        print(f"📊 Loaded {len(bars)} bars | Date range: {self.df.index[0]} → {self.df.index[-1]}")

    def _convert_to_dataframe(self, bars: List[Dict]) -> pd.DataFrame:
        """Convert list of dicts to pandas DataFrame with proper datetime index"""
        df = pd.DataFrame(bars)
        
        # Standardize column names
        col_map = {
            'timestamp': 'datetime',
            'time': 'datetime',
            'date': 'datetime'
        }
        df.rename(columns=col_map, inplace=True)
        
        # Ensure datetime index
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        else:
            # Fallback: assume first column is datetime
            df.index = pd.to_datetime(df.iloc[:, 0])
            df = df.iloc[:, 1:]
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                df[col] = 0.0 if col != 'volume' else 0
        
        df = df[required].astype(float)
        return df

    def step(self) -> Optional[Dict]:
        """Process next bar and return analysis result"""
        if self.df is None or self.cursor >= len(self.df):
            return None

        # Take bars up to current cursor (growing window)
        current_df = self.df.iloc[:self.cursor + 1]

        # Run the full SMK analysis
        try:
            result = self.kernel.on_new_bar(current_df)
            
            if result:
                # Enrich result with bar data for frontend
                current_bar = self.raw_bars[self.cursor]
                enriched = {
                    "bar": {
                        "timestamp": current_bar.get("datetime") or current_bar.get("timestamp"),
                        "open": current_bar.get("open"),
                        "high": current_bar.get("high"),
                        "low": current_bar.get("low"),
                        "close": current_bar.get("close"),
                        "volume": current_bar.get("volume")
                    },
                    "analysis": result
                }
                self.last_result = enriched
                self.cursor += 1
                return enriched

        except Exception as e:
            print(f"❌ Error processing bar {self.cursor}: {e}")
            self.cursor += 1
            return {"error": str(e), "bar_index": self.cursor}

        return None

    def reset_cursor(self):
        """Reset streaming position"""
        self.cursor = 0
        self.running = False
        self.last_result = None
        print("🔄 Pipeline cursor reset")

    def get_status(self) -> Dict:
        """Return current pipeline status"""
        return {
            "total_bars": len(self.raw_bars),
            "processed_bars": self.cursor,
            "is_running": self.running,
            "last_analysis": self.last_result,
            "current_time": datetime.now().isoformat(),
            "has_data": self.df is not None
        }

    def get_latest_analysis(self) -> Optional[Dict]:
        """Return the most recent analysis result"""
        return self.last_result


# For backward compatibility / quick testing
if __name__ == "__main__":
    pipeline = SMKPipeline()
    print("SMK Pipeline ready for FastAPI integration.")
