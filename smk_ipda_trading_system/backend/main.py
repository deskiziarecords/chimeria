"""QUIMERIA / SMK Backend"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn, asyncio, json, os, traceback
import numpy as np
from typing import Optional
from pydantic import BaseModel

app = FastAPI(title="QUIMERIA SMK API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        # Use .item() which always returns a plain Python scalar for any numpy type
        try:
            import numpy as np
            if isinstance(obj, np.generic): return obj.item()
            if isinstance(obj, np.ndarray): return obj.tolist()
        except Exception:
            pass
        t = type(obj).__name__
        if t.startswith('int'):    return int(obj)
        if t.startswith('float'):  return float(obj)
        if t in ('bool_','bool8'): return bool(obj)
        try: return float(obj)
        except Exception: pass
        return super().default(obj)

def _j(data):
    return _SafeEncoder(separators=(',', ':')).encode(data)

@app.exception_handler(Exception)
async def _err(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print(f"[ERROR] {request.url}\n{tb}")
    return JSONResponse(status_code=500,
        content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"})

_here = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.normpath(os.path.join(_here, "..", "frontend"))
if os.path.isdir(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

@app.get("/")
def root():
    idx = os.path.join(FRONTEND, "index.html")
    return FileResponse(idx) if os.path.exists(idx) else {"status": "SMK API running"}

# ── PIPELINE ──────────────────────────────────────────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from smk_pipeline import SMKPipeline
        _pipeline = SMKPipeline()
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

class ModuleConfig(BaseModel):
    disabled_modules: list = []

# ── REST ──────────────────────────────────────────────────────────────────────
@app.post("/api/load/csv")
async def load_csv(payload: CSVPayload):
    from data_connectors import load_csv_text
    bars = load_csv_text(payload.text)
    if not bars:
        raise HTTPException(400, "CSV parse failed")
    get_pipeline().load_bars(bars)
    return {"status": "ok", "count": len(bars), "source": payload.filename}

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
    from data_connectors import generate_sample
    bars = generate_sample(300)
    get_pipeline().load_bars(bars)
    return {"status": "ok", "count": len(bars), "source": "SAMPLE:EURUSD-5M"}

@app.post("/api/config/modules")
def config_modules(payload: ModuleConfig):
    p = get_pipeline()
    disabled = set(payload.disabled_modules)
    for key in list(p.modules.keys()):
        if key.startswith("_"): continue
        if key in disabled:
            if p.modules[key] is not None:
                p.modules["_dis_" + key] = p.modules[key]
                p.modules[key] = None
        else:
            if p.modules.get(key) is None and "_dis_" + key in p.modules:
                p.modules[key] = p.modules.pop("_dis_" + key)
    enabled = [k for k, v in p.modules.items() if v is not None and not k.startswith("_")]
    return {"status": "ok", "enabled": enabled, "disabled": list(disabled)}

@app.get("/api/status")
def get_status():
    return get_pipeline().get_status()

@app.get("/api/ping")
def ping():
    return {"status": "ok", "pipeline_ready": _pipeline is not None}

# ── WEBSOCKET ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()

    # Init pipeline — send error but KEEP socket open on failure
    try:
        p = get_pipeline()
    except Exception as e:
        await ws.send_text(_j({"type": "error", "message": str(e)}))
        await ws.close()
        return

    p.running = False
    run_task: Optional[asyncio.Task] = None
    loop = asyncio.get_event_loop()

    async def run_loop(speed_ms: int):
        try:
            while p.running:
                try:
                    result = await loop.run_in_executor(None, p.step)
                except Exception as e:
                    print(f"[STEP ERR] {e}")
                    traceback.print_exc()
                    try: await ws.send_text(_j({"type": "error", "message": str(e)}))
                    except Exception: pass
                    p.running = False
                    break

                if result is None:
                    try: await ws.send_text(_j({"type": "done"}))
                    except Exception: pass
                    p.running = False
                    break

                try:
                    await ws.send_text(_j({"type": "bar", "data": result}))
                except Exception as e:
                    print(f"[SEND] connection closed: {e}")
                    p.running = False
                    break

                await asyncio.sleep(max(0.016, speed_ms / 1000.0))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[RUN LOOP] {e}")
            traceback.print_exc()
        finally:
            p.running = False

    async def cancel_run():
        nonlocal run_task
        if run_task and not run_task.done():
            p.running = False
            run_task.cancel()
            try: await run_task
            except asyncio.CancelledError: pass
        run_task = None

    try:
        async for raw in ws.iter_text():
            # Parse — skip malformed
            try:
                cmd = json.loads(raw)
            except Exception:
                continue

            action = cmd.get("action", "")

            # Each action is wrapped so errors never close the connection
            try:
                if action == "run":
                    await cancel_run()
                    speed_ms = max(16, int(cmd.get("speed") or 300))
                    p.running = True
                    run_task = asyncio.create_task(run_loop(speed_ms))

                elif action == "step":
                    await cancel_run()
                    result = await loop.run_in_executor(None, p.step)
                    await ws.send_text(_j({"type": "bar", "data": result}
                                          if result else {"type": "done"}))

                elif action == "stop":
                    await cancel_run()
                    await ws.send_text(_j({"type": "stopped"}))

                elif action == "reset":
                    await cancel_run()
                    p.reset_cursor()
                    await ws.send_text(_j({"type": "reset"}))

            except Exception as cmd_err:
                # Bad command — log and continue, do NOT close connection
                print(f"[CMD '{action}'] {cmd_err}")
                traceback.print_exc()
                try: await ws.send_text(_j({"type": "error", "message": str(cmd_err)}))
                except Exception: pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] {e}")
    finally:
        p.running = False
        if run_task and not run_task.done():
            run_task.cancel()
            try: await run_task
            except asyncio.CancelledError: pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
