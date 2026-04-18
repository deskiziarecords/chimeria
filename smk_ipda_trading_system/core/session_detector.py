"""
pandas: For managing time-series OHLCV data and index-based time filtering.
pytz / datetime: For precise EST/EDT timezone synchronization to ensure alignment with New York institutional servers.
dataclasses: For structured session telemetry.
"""
import pandas as pd
from datetime import time
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class SessionStatus:
    is_active: bool
    session_name: Optional[str]
    is_killzone: bool
    temporal_efficiency_score: float # λ2 metric
    status: str

class SessionKillZoneDetector:
    """
    IPDA Layer 1 Component: Temporal Alignment Engine.
    Detects institutional windows (Asian, London, NY) for λ2 validation.
    """
    def __init__(self, timezone: str = 'US/Eastern'):
        self.tz = timezone
        # Defined Ranges based on IPDA Framework [9, 11]
        self.sessions = {
            "ASIAN_RANGE": {"start": time(20, 0), "end": time(2, 0), "killzone": False},
            "LONDON_KILLZONE": {"start": time(2, 0), "end": time(6, 0), "killzone": True},
            "NY_KILLZONE": {"start": time(7, 0), "end": time(11, 0), "killzone": True}
        }

    def _is_time_in_range(self, current: time, start: time, end: time) -> bool:
        """Handles wrap-around time ranges (e.g., 8PM - 2AM)."""
        if start <= end:
            return start <= current <= end
        else: # Over midnight
            return current >= start or current <= end

    def detect_session(self, timestamp: pd.Timestamp) -> SessionStatus:
        """
        Master Logic: Maps current timestamp to institutional delivery windows.
        Calculates λ2 Temporal Alignment [1, 12].
        """
        # Ensure timestamp is in the correct institutional timezone (EST/EDT)
        est_time = timestamp.tz_convert(self.tz).time()
        
        active_session = None
        is_killzone = False
        
        for name, window in self.sessions.items():
            if self._is_time_in_range(est_time, window['start'], window['end']):
                active_session = name
                is_killzone = window['killzone']
                break
        
        # λ2: Temporal Alignment Scoring
        # High score during Killzones, low score during "Dead Zones" [5, 6]
        score = 1.0 if is_killzone else (0.5 if active_session else 0.1)
        
        status_msg = f"ACTIVE: {active_session}" if active_session else "STASIS: DEAD_ZONE"
        if is_killzone: status_msg += " (KILLZONE_VALIDATED)"

        return SessionStatus(
            is_active=active_session is not None,
            session_name=active_session,
            is_killzone=is_killzone,
            temporal_efficiency_score=score,
            status=status_msg
        )

