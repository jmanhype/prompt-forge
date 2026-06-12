"""Prompt Forge — FastAPI server entry point."""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from .config import config
from .forge.engine import ForgeEngine, ForgeEvent
from .forge.session import sessions
from .analyzer.florence import FlorenceAnalyzer
from .analyzer.style import StyleExtractor
from .analyzer.palette import extract_palette

app = FastAPI(title="Prompt Forge", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: ForgeEngine = None


@app.on_event("startup")
async def startup():
    global engine
    config.ensure_dirs()
    engine = ForgeEngine()
    await engine.initialize()


# ─── Static files (frontend) ───
frontend_dir = config.FRONTEND_DIR
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
async def index():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Prompt Forge v0.1.0 — frontend not found. Check /docs for API."}


# ─── Analyze endpoint ───
@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(None), text: str = Form(None)):
    """Analyze an image or text description into structured composition data."""
    if file:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        analyzer = FlorenceAnalyzer(model_id=config.FLORENCE_MODEL)
        result = analyzer.analyze(image)
        prompt_data = analyzer.to_ideogram_json(result)
        prompt_data["palette"] = extract_palette(image)
        
        # Extract style
        style_ext = StyleExtractor()
        prompt_data["style_description"] = style_ext.extract(image)
        
        return {
            "mode": "image",
            "caption": result.caption,
            "background": result.background,
            "palette": prompt_data["palette"],
            "elements": [
                {
                    "id": e.id,
                    "type": e.type,
                    "label": e.label,
                    "description": e.description,
                    "bbox": e.bbox,
                    "confidence": e.confidence,
                }
                for e in result.elements
            ],
            "json": prompt_data,
            "model": result.model_used,
        }
    
    elif text:
        # Text mode
        return {
            "mode": "text",
            "caption": text,
            "elements": [],
            "json": {
                "caption": text,
                "composition": {"background": text, "elements": []},
                "style_description": {},
            },
        }
    
    raise HTTPException(400, "Provide either 'file' or 'text'")


# ─── Forge endpoint (SSE stream) ───
@app.post("/api/forge")
async def start_forge(
    description: str = Form(""),
    file: UploadFile = File(None),
    max_iterations: int = Form(None),
    threshold: float = Form(None),
):
    """Start a forge session. Returns session_id for WebSocket connection."""
    session_id = str(uuid.uuid4())[:8]
    
    image = None
    if file:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    
    session = sessions.create(session_id, description)
    session.status = "running"
    
    # Run forge in background
    asyncio.create_task(_run_forge(session_id, description, image, max_iterations, threshold))
    
    return {"session_id": session_id, "ws_url": f"/ws/forge/{session_id}"}


async def _run_forge(session_id: str, description: str, image, max_iterations, threshold):
    """Background task that runs the forge and pushes events to session."""
    session = sessions.get(session_id)
    if not session:
        return
    
    try:
        async for event in engine.run(description, image, max_iterations, threshold):
            await session.add_event(event)
            if event.type == "converged":
                session.status = "converged"
            elif event.type == "error":
                session.status = "error"
    except Exception as e:
        await session.add_event(ForgeEvent(type="error", data={"message": str(e)}))
        session.status = "error"


# ─── WebSocket for live forge updates ───
@app.websocket("/ws/forge/{session_id}")
async def forge_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = sessions.get(session_id)
    
    if not session:
        await websocket.send_json({"type": "error", "data": {"message": "Session not found"}})
        await websocket.close()
        return
    
    await websocket.send_json({"type": "connected", "data": {"session_id": session_id}})
    
    try:
        while True:
            event = await session.next_event(timeout=10)
            if event is None:
                # Check if session is done
                if session.status in ("converged", "error"):
                    break
                continue
            
            await websocket.send_json(event.type if isinstance(event, dict) else {
                "type": event.type,
                "data": event.data,
            })
            
            if event.type in ("converged", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        sessions.remove(session_id)


# ─── Forge status ───
@app.get("/api/forge/{session_id}")
async def get_forge_status(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session_id,
        "status": session.status,
        "events": session.events,
    }


# ─── Composition Library ───
@app.get("/api/library")
async def list_library(q: str = "", limit: int = 50, offset: int = 0):
    if q:
        return {"results": engine.db.search_compositions(q, limit)}
    return {"results": engine.db.list_compositions(limit, offset)}


@app.get("/api/library/{comp_id}")
async def get_composition(comp_id: str):
    comp = engine.db.get_composition(comp_id)
    if not comp:
        raise HTTPException(404, "Composition not found")
    return comp


@app.get("/api/library/stats")
async def library_stats():
    return engine.db.get_stats()


# ─── LoRA Detection ───
@app.get("/api/loras")
async def list_loras():
    return {"loras": engine.lora_detector.list_loras()}


@app.post("/api/rescan-loras")
async def rescan_loras():
    loras = engine.lora_detector.scan()
    return {"count": len(loras), "loras": engine.lora_detector.list_loras()}


# ─── Capabilities ───
@app.get("/api/capabilities")
async def get_capabilities():
    from .compiler.capability import probe_capabilities
    caps = await probe_capabilities(config.COMFYUI_URL)
    return {
        "comfyui_connected": caps.connected,
        "strategy": caps.best_strategy,
        "strategy_description": caps.strategy_description,
        "has_gligen": caps.has_gligen,
        "has_attention_couple": caps.has_attention_couple,
        "has_ipadapter_region": caps.has_ipadapter_region,
        "has_flux_guidance": caps.has_flux_guidance,
        "available_checkpoints": caps.available_checkpoints[:10],
        "available_loras": caps.available_loras[:20],
    }


# ─── Image serving ───
@app.get("/api/image/{filename}")
async def serve_image(filename: str):
    path = config.OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(str(path), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.FORGE_PORT)
