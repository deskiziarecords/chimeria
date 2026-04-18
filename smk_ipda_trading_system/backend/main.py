"""
backend/main.py
QUIMERIA / SMK FastAPI Backend - Final Corrected Version
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import asyncio
import json
import os
from typing import Optional

from pydantic import BaseModel

# ==================== CORRECT IMPORTS ====================
# Import from the same backend package (relative imports)
from .smk_pipeline import SMKPipeline
from .data_connectors import (
    load_csv_text,
    fetch_bitget,
    fetch_oanda,
    generate_sample
)
# ========================================================

app = FastAPI(
    title="QUIMERIA SMK API",
    description="Sovereign Market Kernel with IPDA + Lambda Fusion",
    version="1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend if it exists
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(FRONTEND_PATH):
    app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

# Global pipeline
pipeline = SMKPipeline()

# Pydantic Models
class CSVPayload(BaseModel):
    text: str
    filename: Optional[str] = "upload.csv"

class BitgetPayload(BaseModel):
    api_key: str
    api_secret: str
    symbol: str = "BTCUSDT"
    granularity: str = "5m"
    limit: int = 300

class OandaPayload(BaseModel):
    token: str
    account_id: str
    instrument: str = "EUR_USD"
    granularity: str = "M5"
    count: int = 300

# Routes
@app.get("/")
async def root():
    index_path = os.path.join(FRONTEND_PATH, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "QUIMERIA SMK Backend is running"}

@app.post("/api/load/csv")
async def load_csv(payload: CSVPayload):
    try:
        bars = load_csv_text(payload.text)
        if not bars:
            raise HTTPException(status_code=400, detail="Failed to parse CSV")
        pipeline.load_bars(bars)
        return {"status": "ok", "count": len(bars), "source": payload.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading CSV: {str(e)}")

@app.post("/api/load/bitget")
async def load_bitget(payload: BitgetPayload):
    bars = await fetch_bitget(payload.api_key, payload.api_secret, payload.symbol, payload.granularity, payload.limit)
    pipeline.load_bars(bars)
    return {"status": "ok", "count": len(bars), "source": f"BITGET:{payload.symbol}"}

@app.post("/api/load/oanda")
async def load_oanda(payload: OandaPayload):
    bars = await fetch_oanda(payload.token, payload.account_id, payload.instrument, payload.granularity, payload.count)
    pipeline.load_bars(bars)
    return {"status": "ok", "count": len(bars), "source": f"OANDA:{payload.instrument}"}

@app.post("/api/load/sample")
async def load_sample():
    bars = generate_sample(300)
    pipeline.load_bars(bars)
    return {"status": "ok", "count": len(bars), "source": "SAMPLE"}

@app.get("/api/bars")
async def get_bars():
    return {"bars": pipeline.raw_bars[:100]}

@app.get("/api/status")
async def get_status():
    return pipeline.get_status()

# WebSocket Streaming
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    pipeline.reset_cursor()
    try:
        async for message in websocket.iter_text():
            cmd = json.loads(message)
            action = cmd.get("action")

            if action == "step":
                result = pipeline.step()
                await websocket.send_json({"type": "bar", "data": result} if result else {"type": "done"})
            elif action == "run":
                speed = cmd.get("speed", 300)
                pipeline.running = True
                while pipeline.running:
                    result = pipeline.step()
                    if result:
                        await websocket.send_json({"type": "bar", "data": result})
                        await asyncio.sleep(speed / 1000)
                    else:
                        await websocket.send_json({"type": "done"})
                        break
            elif action == "stop":
                pipeline.running = False
                await websocket.send_json({"type": "stopped"})
            elif action == "reset":
                pipeline.reset_cursor()
                await websocket.send_json({"type": "reset"})
    except WebSocketDisconnect:
        pipeline.running = False
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
