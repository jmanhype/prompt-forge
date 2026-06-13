"""FastAPI server for Prompt Forge v2."""
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

from .config import Config

app = FastAPI(title="Prompt Forge v2", version="2.0.0")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load system prompt
SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text().strip()

# Mount static files
app.mount("/outputs", StaticFiles(directory=Config.OUTPUTS_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=Config.FRONTEND_DIR), name="static")


class GenerateRequest(BaseModel):
    description: str
    lora: str = "ektachrome_style_v1.safetensors"
    lora_strength: float = 0.8


@app.get("/")
async def root():
    """Serve the frontend."""
    return FileResponse(Config.FRONTEND_DIR / "index.html")


@app.get("/health")
async def health():
    """Check connectivity to ComfyUI and LLM."""
    comfyui_ok = False
    llm_ok = False
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{Config.COMFYUI_URL}/system_stats")
            comfyui_ok = resp.status_code == 200
    except:
        pass
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{Config.LLM_URL}/models", 
                                   headers={"Authorization": f"Bearer {Config.LLM_API_KEY}"})
            llm_ok = resp.status_code == 200
    except:
        pass
    
    return {
        "comfyui": comfyui_ok,
        "llm": llm_ok,
        "comfyui_url": Config.COMFYUI_URL,
        "llm_url": Config.LLM_URL
    }


@app.get("/system-prompt")
async def get_system_prompt():
    """Return the system prompt."""
    return {"system_prompt": SYSTEM_PROMPT}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate an image from description using system prompt + Ektachrome LoRA."""
    
    # Step 1: Generate JSON prompt using system prompt
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            llm_response = await client.post(
                f"{Config.LLM_URL}/chat/completions",
                headers={"Authorization": f"Bearer {Config.LLM_API_KEY}"},
                json={
                    "model": Config.LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": req.description}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            )
            llm_response.raise_for_status()
            llm_data = llm_response.json()
            json_prompt_text = llm_data["choices"][0]["message"]["content"]
            
            # Parse the JSON (handle markdown code blocks)
            if "```json" in json_prompt_text:
                json_prompt_text = json_prompt_text.split("```json")[1].split("```")[0]
            elif "```" in json_prompt_text:
                json_prompt_text = json_prompt_text.split("```")[1].split("```")[0]
            
            json_prompt = json.loads(json_prompt_text.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")
    
    # Step 2: Build ComfyUI workflow
    prompt_string = json.dumps(json_prompt, ensure_ascii=False)
    
    workflow = {
        "prompt": {
            "1": {"class_type": "UNETLoader", "inputs": {
                "unet_name": "ideogram4_fp8_scaled.safetensors",
                "weight_dtype": "fp8_e4m3fn"
            }},
            "2": {"class_type": "CLIPLoader", "inputs": {
                "clip_name": "qwen3vl_8b_fp8_scaled.safetensors",
                "type": "ideogram4"
            }},
            "3": {"class_type": "LoraLoader", "inputs": {
                "model": ["1", 0],
                "clip": ["2", 0],
                "lora_name": req.lora,
                "strength_model": req.lora_strength,
                "strength_clip": req.lora_strength
            }},
            "4": {"class_type": "CLIPTextEncode", "inputs": {
                "clip": ["3", 1],
                "text": prompt_string
            }},
            "5": {"class_type": "ConditioningZeroOut", "inputs": {
                "conditioning": ["4", 0]
            }},
            "6": {"class_type": "ModelSamplingAuraFlow", "inputs": {
                "model": ["3", 0],
                "shift": 5.0
            }},
            "7": {"class_type": "BasicGuider", "inputs": {
                "model": ["6", 0],
                "conditioning": ["4", 0]
            }},
            "8": {"class_type": "KSamplerSelect", "inputs": {
                "sampler_name": "euler"
            }},
            "9": {"class_type": "BasicScheduler", "inputs": {
                "model": ["6", 0],
                "scheduler": "simple",
                "steps": 30,
                "denoise": 1.0
            }},
            "10": {"class_type": "EmptySD3LatentImage", "inputs": {
                "width": 1024,
                "height": 768,
                "batch_size": 1
            }},
            "11": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["12", 0],
                "guider": ["7", 0],
                "sampler": ["8", 0],
                "sigmas": ["9", 0],
                "latent_image": ["10", 0]
            }},
            "12": {"class_type": "RandomNoise", "inputs": {
                "noise_seed": 42
            }},
            "13": {"class_type": "VAELoader", "inputs": {
                "vae_name": "flux2-vae.safetensors"
            }},
            "14": {"class_type": "VAEDecode", "inputs": {
                "samples": ["11", 0],
                "vae": ["13", 0]
            }},
            "15": {"class_type": "SaveImage", "inputs": {
                "filename_prefix": "forge_v2",
                "images": ["14", 0]
            }}
        },
        "client_id": "prompt-forge-v2"
    }
    
    # Step 3: Submit to ComfyUI
    import time
    import uuid
    
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{Config.COMFYUI_URL}/prompt", json=workflow)
            resp.raise_for_status()
            prompt_data = resp.json()
            prompt_id = prompt_data.get("prompt_id")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ComfyUI submission failed: {str(e)}")
    
    # Step 4: Poll for completion
    max_wait = 120  # 2 minutes
    elapsed = 0
    
    while elapsed < max_wait:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                hist_resp = await client.get(f"{Config.COMFYUI_URL}/history/{prompt_id}")
                hist_resp.raise_for_status()
                history = hist_resp.json()
                
                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status", {})
                    
                    if status.get("completed"):
                        outputs = entry.get("outputs", {})
                        for node_id, output in outputs.items():
                            if "images" in output:
                                image_data = output["images"][0]
                                filename = image_data["filename"]
                                
                                # Download from ComfyUI and save to outputs
                                dest_path = Config.OUTPUTS_DIR / f"{session_id}.png"
                                
                                async with httpx.AsyncClient(timeout=30.0) as img_client:
                                    img_resp = await img_client.get(
                                        f"{Config.COMFYUI_URL}/view",
                                        params={"filename": filename}
                                    )
                                    img_resp.raise_for_status()
                                    dest_path.write_bytes(img_resp.content)
                                
                                duration_ms = int((time.time() - start_time) * 1000)
                                
                                return {
                                    "session_id": session_id,
                                    "description": req.description,
                                    "json_prompt": json_prompt,
                                    "image": f"/outputs/{session_id}.png",
                                    "duration_ms": duration_ms
                                }
        except:
            pass
        
        import asyncio
        await asyncio.sleep(3)
        elapsed += 3
    
    raise HTTPException(status_code=504, detail="Generation timed out")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7861)
