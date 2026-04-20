"""
realtime.py — Live candle streaming via Bitget REST polling + WebSocket broadcast.
Polls Bitget every N seconds, runs SMK detectors, streams to all connected frontends.
"""
import asyncio, json, time, httpx
from typing import Set, Optional, Dict, Any

# ── BITGET CANDLE GRANULARITIES ───────────────────────────────────────────────
GRAN_MAP = {
    "1m": ("1min",  60),
    "3m": ("3min",  180),
    "5m": ("5min",  300),
    "15m":("15min", 900),
    "1h": ("1H",    3600),
}

class LiveFeed:
    """
    Polls Bitget REST for the latest closed candle, runs the SMK pipeline,
    broadcasts bar payloads to all subscribed WebSocket clients.
    """
    def __init__(self):
        self.clients:    Set[Any]  = set()   # WebSocket connections
        self.running:    bool      = False
        self.symbol:     str       = "EURUSDT"
        self.granularity:str       = "1m"
        self.api_key:    str       = ""
        self.last_ts:    int       = 0       # timestamp of last processed candle
        self.pipeline:   Any       = None    # SMKPipeline instance
        self.task:       Optional[asyncio.Task] = None

    def configure(self, symbol: str, granularity: str, api_key: str, pipeline):
        self.symbol      = symbol.upper()
        self.granularity = granularity
        self.api_key     = api_key
        self.pipeline    = pipeline
        self.last_ts     = 0

    def add_client(self, ws):
        self.clients.add(ws)

    def remove_client(self, ws):
        self.clients.discard(ws)

    async def broadcast(self, msg: dict):
        dead = set()
        payload = json.dumps(msg, separators=(',', ':'))
        for ws in list(self.clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self.clients -= dead

    async def fetch_candles(self, limit: int = 100):
        gran_str, _ = GRAN_MAP.get(self.granularity, ("1min", 60))
        url = "https://api.bitget.com/api/v2/spot/market/candles"
        params = {
            "symbol":      self.symbol,
            "granularity": gran_str,
            "limit":       str(limit),
        }
        headers = {}
        if self.api_key:
            headers["ACCESS-KEY"] = self.api_key

        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            data = r.json()

        if data.get("code") != "00000":
            raise ValueError(f"Bitget: {data.get('msg','error')}")

        bars = []
        for c in data.get("data", []):
            ts = int(c[0]) // 1000  # ms → seconds
            bars.append({
                "time":   ts,
                "open":   float(c[1]),
                "high":   float(c[2]),
                "low":    float(c[3]),
                "close":  float(c[4]),
                "volume": float(c[5]),
            })
        bars.sort(key=lambda b: b["time"])
        return bars

    async def _bootstrap(self):
        """Load 100 bars of history to warm up the pipeline."""
        try:
            bars = await self.fetch_candles(limit=100)
            if bars:
                self.pipeline.load_bars(bars)
                self.last_ts = bars[-1]["time"]
                print(f"[LIVE] Bootstrapped {len(bars)} bars, last ts={self.last_ts}")
                return True
        except Exception as e:
            print(f"[LIVE] Bootstrap error: {e}")
        return False

    async def run(self):
        """Main polling loop — checks for new closed candle every ~5s."""
        self.running = True
        _, poll_interval = GRAN_MAP.get(self.granularity, ("1min", 60))
        # Poll every 5 seconds (check if a new candle has closed)
        check_interval = 5

        print(f"[LIVE] Starting feed: {self.symbol} {self.granularity} poll={check_interval}s")

        # Bootstrap
        ok = await self._bootstrap()
        if not ok:
            await self.broadcast({"type": "error", "message": "Failed to fetch initial candles from Bitget"})
            self.running = False
            return

        while self.running:
            await asyncio.sleep(check_interval)
            if not self.clients:
                continue  # no one listening, skip

            try:
                bars = await self.fetch_candles(limit=5)
                if not bars:
                    continue

                # Find candles newer than last processed
                new_bars = [b for b in bars if b["time"] > self.last_ts]

                for bar in new_bars:
                    # Skip the very latest candle — it may still be open/forming
                    # Only process candles that are at least poll_interval seconds old
                    age = int(time.time()) - bar["time"]
                    if age < poll_interval - 2:
                        continue  # candle still forming

                    # Append to pipeline history
                    self.pipeline.raw_bars.append(bar)
                    self.pipeline.cursor = len(self.pipeline.raw_bars) - 1

                    # Run SMK detectors
                    try:
                        result = self.pipeline.step()
                        if result:
                            result["live"] = True
                            result["symbol"] = self.symbol
                            await self.broadcast({"type": "bar", "data": result})
                    except Exception as step_err:
                        print(f"[LIVE] Step error: {step_err}")

                    self.last_ts = bar["time"]

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[LIVE] Poll error: {e}")
                await self.broadcast({"type": "error", "message": f"Feed error: {e}"})
                await asyncio.sleep(15)  # backoff

        self.running = False
        print("[LIVE] Feed stopped")

    def start(self, pipeline):
        self.pipeline = pipeline
        if self.task and not self.task.done():
            return
        self.task = asyncio.create_task(self.run())

    def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()


# Global singleton
live_feed = LiveFeed()
