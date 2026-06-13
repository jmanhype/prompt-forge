"""Prompt Forge API — simple server using proven system prompt + Ektachrome LoRA."""
import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from server.forge.simple import SimpleForge
from server.config import config

app = FastAPI(title="Prompt Forge", version="2.0.0")

# Ensure dirs exist
config.ensure_dirs()

# Serve outputs as static files
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUTS_DIR)), name="outputs")

forge = SimpleForge()


class GenerateRequest(BaseModel):
    description: str
    lora: str = "ektachrome_style_v1.safetensors"
    lora_strength: float = 0.8


class GenerateResponse(BaseModel):
    session_id: str
    description: str
    json_prompt: dict
    image: str  # path or URL
    duration_ms: int


@app.get("/")
async def root():
    return {"status": "ok", "version": "2.0.0", "docs": "/docs"}


@app.get("/health")
async def health():
    """Check ComfyUI and LLM connectivity."""
    import httpx

    results = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        # Check ComfyUI
        try:
            resp = await client.get(f"{config.COMFYUI_URL}/system_stats")
            results["comfyui"] = "ok" if resp.status_code == 200 else f"error: {resp.status_code}"
        except Exception as e:
            results["comfyui"] = f"error: {str(e)}"

        # Check LLM
        try:
            resp = await client.get(f"{config.LLM_URL}/v1/models")
            results["llm"] = "ok" if resp.status_code == 200 else f"error: {resp.status_code}"
        except Exception as e:
            results["llm"] = f"error: {str(e)}"

    return results


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate an image from a text description.
    
    Pipeline: description → system prompt → Ideogram 4 JSON → ComfyUI → image
    """
    try:
        # Update forge settings if custom
        forge.lora_name = req.lora
        forge.lora_strength = req.lora_strength

        result = await forge.generate(req.description)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/image/{filename}")
async def get_image(filename: str):
    """Serve a generated image."""
    path = config.OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png")


@app.get("/system-prompt")
async def get_system_prompt():
    """Return the system prompt being used."""
    return {"system_prompt": forge.system_prompt}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.FORGE_PORT)
