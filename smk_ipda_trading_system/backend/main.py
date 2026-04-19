"""
QUIMERIA / SMK Backend
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn, asyncio, json, os, traceback
from typing import Optional
from pydantic import BaseModel

app = FastAPI(title="QUIMERIA SMK API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print(f"[ERROR] {request.url}\n{tb}")
    return JSONResponse(status_code=500,
        content={"detail": str(exc), "traceback": tb[-2000:]},
        headers={"Access-Control-Allow-Origin": "*"})

# ── FRONTEND ──────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.normpath(os.path.join(_here, "..", "frontend"))
if os.path.isdir(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

@app.get("/")
def root():
    idx = os.path.join(FRONTEND, "index.html")
    return FileResponse(idx) if os.path.exists(idx) else {"status": "QUIMERIA SMK API running"}

# ── PIPELINE ──────────────────────────────────────────────────────────────────
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
    api_key: str; api_secret: str; symbol: str = "EURUSDT"
    granularity: str = "5m"; limit: int = 300

class OandaPayload(BaseModel):
    token: str; account_id: str; instrument: str = "EUR_USD"
    granularity: str = "M5"; count: int = 300

# ── REST ──────────────────────────────────────────────────────────────────────
@app.post("/api/load/csv")
async def load_csv(payload: CSVPayload):
    try:
        from data_connectors import load_csv_text
        bars = load_csv_text(payload.text)
        if not bars:
            raise HTTPException(400, "CSV parse failed — need: datetime,open,high,low,close,volume")
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

@app.get("/api/status")
def get_status():
    return get_pipeline().get_status()

@app.get("/api/ping")
def ping():
    return {"status": "ok", "pipeline_ready": _pipeline is not None,
            "pipeline_error": _pipeline_error}

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
    p.running = False
    run_task: Optional[asyncio.Task] = None

    async def run_loop(speed_ms: int):
        """Runs bars continuously, yields to event loop between each bar."""
        try:
            while p.running:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, p.step          # run step() in thread so it can't block loop
                )
                if result is None:
                    await ws.send_json({"type": "done"})
                    p.running = False
                    break
                try:
                    await ws.send_json({"type": "bar", "data": result})
                except Exception:
                    p.running = False
                    break
                # Yield to event loop so stop/reset messages can be received
                await asyncio.sleep(speed_ms / 1000.0)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[RUN LOOP] {e}")
            traceback.print_exc()
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            p.running = False

    try:
        async for raw in ws.iter_text():
            try:
                cmd = json.loads(raw)
            except Exception:
                continue

            action = cmd.get("action")

            if action == "run":
                # Cancel any existing run
                if run_task and not run_task.done():
                    run_task.cancel()
                    try:
                        await run_task
                    except asyncio.CancelledError:
                        pass
                speed_ms = max(16, int(cmd.get("speed", 300)))
                p.running = True
                run_task = asyncio.create_task(run_loop(speed_ms))

            elif action == "step":
                # Cancel running if active
                if run_task and not run_task.done():
                    p.running = False
                    run_task.cancel()
                    try:
                        await run_task
                    except asyncio.CancelledError:
                        pass
                result = await asyncio.get_event_loop().run_in_executor(None, p.step)
                await ws.send_json({"type": "bar", "data": result}
                                   if result else {"type": "done"})

            elif action == "stop":
                p.running = False
                if run_task and not run_task.done():
                    run_task.cancel()
                    try:
                        await run_task
                    except asyncio.CancelledError:
                        pass
                await ws.send_json({"type": "stopped"})

            elif action == "reset":
                p.running = False
                if run_task and not run_task.done():
                    run_task.cancel()
                    try:
                        await run_task
                    except asyncio.CancelledError:
                        pass
                p.reset_cursor()
                await ws.send_json({"type": "reset"})

    except WebSocketDisconnect:
        p.running = False
        if run_task and not run_task.done():
            run_task.cancel()
    except Exception as e:
        print(f"[WS ERROR] {e}\n{traceback.format_exc()}")
        p.running = False
        if run_task and not run_task.done():
            run_task.cancel()
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
