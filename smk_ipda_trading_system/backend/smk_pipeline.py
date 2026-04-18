"""
backend/smk_pipeline.py
SMK Pipeline - Orchestrates the SovereignMarketKernel for the FastAPI backend
"""

import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime

# Correct import from root
from SovereignMarketKernel import SovereignMarketKernel


class SMKPipeline:
    """
    High-level pipeline that wraps the SovereignMarketKernel
    for easy use in FastAPI.
    """

    def __init__(self):
        self.kernel = SovereignMarketKernel()
        self.raw_bars: List[Dict] = []
        self.df: Optional[pd.DataFrame] = None
        self.cursor = 0
        self.running = False
        self.last_result: Optional[Dict] = None

        print("✅ SMK Pipeline initialized")

    def load_bars(self, bars: List[Dict]):
        """Load new bars from any source"""
        if not bars:
            return

        self.raw_bars = bars
        self.df = self._convert_to_dataframe(bars)
        self.cursor = 0
        self.last_result = None

        print(f"📊 Loaded {len(bars)} bars | Range: {self.df.index[0]} → {self.df.index[-1]}")

    def _convert_to_dataframe(self, bars: List[Dict]) -> pd.DataFrame:
        """Convert list of dicts → pandas DataFrame with datetime index"""
        df = pd.DataFrame(bars)
        
        # Standardize datetime column
        for col in ['timestamp', 'time', 'datetime', 'date']:
            if col in df.columns:
                df['datetime'] = pd.to_datetime(df[col])
                break
        else:
            # Assume first column is datetime if none found
            df['datetime'] = pd.to_datetime(df.iloc[:, 0])
        
        df.set_index('datetime', inplace=True)
        
        # Ensure required columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                df[col] = 0.0 if col != 'volume' else 0.0
        
        return df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    def step(self) -> Optional[Dict]:
        """Process next bar"""
        if self.df is None or self.cursor >= len(self.df):
            return None

        current_df = self.df.iloc[:self.cursor + 1]

        try:
            result = self.kernel.on_new_bar(current_df)
            
            if result:
                current_bar = self.raw_bars[self.cursor]
                enriched = {
                    "bar": {
                        "timestamp": current_bar.get("timestamp") or current_bar.get("datetime"),
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
            return {"error": str(e)}

        return None

    def reset_cursor(self):
        self.cursor = 0
        self.running = False
        self.last_result = None
        print("🔄 Pipeline cursor reset")

    def get_status(self) -> Dict:
        return {
            "total_bars": len(self.raw_bars),
            "processed_bars": self.cursor,
            "is_running": self.running,
            "has_data": self.df is not None
        }


if __name__ == "__main__":
    print("SMK Pipeline ready.")
