"""
QUIMERIA / SMK Backend — FastAPI
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn, asyncio, json, os, traceback
from typing import Optional
from pydantic import BaseModel

app = FastAPI(title="QUIMERIA SMK API")

# CORS — must be added BEFORE any routes, allow everything for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler — ensures CORS headers survive 500s
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print(f"[ERROR] {request.url}\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": tb[-2000:]},
        headers={"Access-Control-Allow-Origin": "*"},
    )

# ── FRONTEND ──────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.normpath(os.path.join(_here, "..", "frontend"))

if os.path.isdir(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

@app.get("/")
def root():
    idx = os.path.join(FRONTEND, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"status": "QUIMERIA SMK API running", "frontend": FRONTEND}

# ── PIPELINE (lazy, crash-isolated) ──────────────────────────────────────────
_pipeline = None
_pipeline_error = None

def get_pipeline():
    global _pipeline, _pipeline_error
    if _pipeline is None:
        try:
            from smk_pipeline import SMKPipeline
            _pipeline = SMKPipeline()
            _pipeline_error = None
        except Exception as e:
            _pipeline_error = str(e)
            traceback.print_exc()
            raise HTTPException(500, f"Pipeline init failed: {e}")
    return _pipeline

# ── MODELS ────────────────────────────────────────────────────────────────────
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

# ── REST ──────────────────────────────────────────────────────────────────────
@app.post("/api/load/csv")
async def load_csv(payload: CSVPayload):
    try:
        from data_connectors import load_csv_text
        bars = load_csv_text(payload.text)
        if not bars:
            raise HTTPException(400, "CSV parse failed — need columns: datetime,open,high,low,close,volume")
        get_pipeline().load_bars(bars)
        return {"status": "ok", "count": len(bars), "source": payload.filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/load/bitget")
async def load_bitget(payload: BitgetPayload):
    from data_connectors import fetch_bitget
    bars = await fetch_bitget(payload.api_key, payload.api_secret,
                               payload.symbol, payload.granularity, payload.limit)
    get_pipeline().load_bars(bars)
    return {"status": "ok", "count": len(bars), "source": f"BITGET:{payload.symbol}"}

@app.post("/api/load/oanda")
async def load_oanda(payload: OandaPayload):
    from data_connectors import fetch_oanda
    bars = await fetch_oanda(payload.token, payload.account_id,
                              payload.instrument, payload.granularity, payload.count)
    get_pipeline().load_bars(bars)
    return {"status": "ok", "count": len(bars), "source": f"OANDA:{payload.instrument}"}

@app.post("/api/load/sample")
async def load_sample():
    try:
        from data_connectors import generate_sample
        bars = generate_sample(300)
        get_pipeline().load_bars(bars)
        return {"status": "ok", "count": len(bars), "source": "SAMPLE:EURUSD-5M"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/bars")
def get_bars():
    return {"bars": get_pipeline().raw_bars[:100]}

@app.get("/api/status")
def get_status():
    p = get_pipeline()
    return p.get_status()

@app.get("/api/ping")
def ping():
    """Health check — use this to verify server is up."""
    return {
        "status": "ok",
        "pipeline_ready": _pipeline is not None,
        "pipeline_error": _pipeline_error,
    }

# ── WEBSOCKET ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        p = get_pipeline()
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
        return

    p.reset_cursor()
    try:
        async for msg in ws.iter_text():
            try:
                cmd = json.loads(msg)
            except Exception:
                continue
            action = cmd.get("action")

            if action == "step":
                result = p.step()
                await ws.send_json({"type": "bar", "data": result}
                                   if result else {"type": "done"})

            elif action == "run":
                speed_ms = max(16, int(cmd.get("speed", 300)))
                p.running = True
                while p.running:
                    result = p.step()
                    if result:
                        await ws.send_json({"type": "bar", "data": result})
                        await asyncio.sleep(speed_ms / 1000.0)
                    else:
                        await ws.send_json({"type": "done"})
                        p.running = False
                        break

            elif action == "stop":
                p.running = False
                await ws.send_json({"type": "stopped"})

            elif action == "reset":
                p.reset_cursor()
                await ws.send_json({"type": "reset"})

    except WebSocketDisconnect:
        p.running = False
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[WS ERROR] {e}\n{tb}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
