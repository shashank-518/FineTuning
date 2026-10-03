"""Badminton coach web UI — FastAPI + static frontend."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
sys.path.insert(0, str(ROOT / "scripts"))

from runtime import BadmintonModel, load_config  # noqa: E402

_model: BadmintonModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    print("Loading badminton LoRA model (first load can take a minute on CPU)...")
    _model = BadmintonModel()
    print(f"Ready on {_model.device} | adapter: {_model.adapter}")
    yield
    _model = None


app = FastAPI(title="Shuttle — Badminton Coach", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health():
    cfg = load_config()
    return {
        "ok": _model is not None,
        "device": getattr(_model, "device", None),
        "model": cfg["model_name"],
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    q = req.message.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty message")
    try:
        reply = _model.ask(q)
    except Exception as exc:  # noqa: BLE001 — surface to UI
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(reply=reply)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=7860,
        reload=False,
        app_dir=str(Path(__file__).resolve().parent),
    )
