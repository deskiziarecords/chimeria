"""
QUIMERIA / SMK Backend — FastAPI
Exposes the full SMK pipeline over HTTP + WebSocket
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn, asyncio, json, os
from typing import Optional
from pydantic import BaseModel

from smk_pipeline import SMKPipeline
from data_connectors import load_csv_text, fetch_bitget, fetch_oanda

app = FastAPI(title="QUIMERIA SMK API")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])

# Serve the frontend
FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND, "index.html"))

# ─── SMK PIPELINE (singleton) ────────────────────────────────────────────────
pipeline = SMKPipeline()

# ─── REST ENDPOINTS ───────────────────────────────────────────────────────────
class CSVPayload(BaseModel):
    text: str
    filename: Optional[str] = "upload.csv"

class BitgetPayload(BaseModel):
    api_key: str
    api_secret: str
    symbol: str = "EURUSDT"
    granularity: str = "5m"
    limit: int = 300

class OandaPayload(BaseModel):
    token: str
    account_id: str
    instrument: str = "EUR_USD"
    granularity: str = "M5"
    count: int = 300

@app.post("/api/load/csv")
async def load_csv(payload: CSVPayload):
    bars = load_csv_text(payload.text)
    if not bars:
        raise HTTPException(400, "CSV parse failed — need datetime,open,high,low,close,volume")
    pipeline.load_bars(bars)
    return {"status": "ok", "count": len(bars), "source": payload.filename}

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
    from data_connectors import generate_sample
    bars = generate_sample(300)
    pipeline.load_bars(bars)
    return {"status": "ok", "count": len(bars), "source": "SAMPLE:EURUSD-5M"}

@app.get("/api/bars")
def get_bars():
    return {"bars": pipeline.raw_bars}

@app.get("/api/status")
def get_status():
    return pipeline.get_status()

# ─── WEBSOCKET — streaming bar-by-bar analysis ───────────────────────────────
@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    pipeline.reset_cursor()
    try:
        async for msg in ws.iter_text():
            cmd = json.loads(msg)
            action = cmd.get("action")

            if action == "step":
                result = pipeline.step()
                if result:
                    await ws.send_json({"type": "bar", "data": result})
                else:
                    await ws.send_json({"type": "done"})

            elif action == "run":
                speed_ms = cmd.get("speed", 300)
                pipeline.running = True
                while pipeline.running:
                    result = pipeline.step()
                    if result:
                        await ws.send_json({"type": "bar", "data": result})
                        await asyncio.sleep(speed_ms / 1000.0)
                    else:
                        await ws.send_json({"type": "done"})
                        pipeline.running = False
                        break

            elif action == "stop":
                pipeline.running = False
                await ws.send_json({"type": "stopped"})

            elif action == "reset":
                pipeline.reset_cursor()
                await ws.send_json({"type": "reset"})

    except WebSocketDisconnect:
        pipeline.running = False

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
