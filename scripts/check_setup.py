#!/usr/bin/env python3
"""Pre-flight checks for Prompt Forge."""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent


def check(name: str, condition: bool, fix: str = ""):
    status = "OK" if condition else "MISSING"
    icon = "✓" if condition else "✗"
    print(f"  {icon} {name}: {status}")
    if not condition and fix:
        print(f"    Fix: {fix}")
    return condition


def main():
    print("\nPrompt Forge — Setup Check\n" + "=" * 40)
    all_ok = True
    
    # Python version
    py_ok = sys.version_info >= (3, 10)
    all_ok &= check(f"Python {sys.version.split()[0]}", py_ok, "Install Python 3.10+")
    
    # .env file
    env_exists = (ROOT / ".env").exists()
    all_ok &= check(".env file", env_exists, "cp .env.example .env && edit")
    
    # Requirements
    try:
        import fastapi
        all_ok &= check("FastAPI", True)
    except ImportError:
        all_ok &= check("FastAPI", False, "pip install -r requirements.txt")
    
    try:
        import torch
        all_ok &= check(f"PyTorch {torch.__version__}", True)
        all_ok &= check(f"CUDA available", torch.cuda.is_available(),
                        "CPU mode works but is slower")
    except ImportError:
        all_ok &= check("PyTorch", False, "pip install torch torchvision")
    
    try:
        import transformers
        all_ok &= check(f"Transformers {transformers.__version__}", True)
    except ImportError:
        all_ok &= check("Transformers", False, "pip install transformers")
    
    try:
        import websockets
        all_ok &= check("WebSockets", True)
    except ImportError:
        all_ok &= check("WebSockets", False, "pip install websockets")
    
    # ComfyUI connectivity
    print()
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv(ROOT / ".env")
        comfyui_url = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
        
        import urllib.request
        resp = urllib.request.urlopen(f"{comfyui_url}/system_stats", timeout=5)
        all_ok &= check(f"ComfyUI at {comfyui_url}", resp.status == 200)
    except Exception as e:
        all_ok &= check(f"ComfyUI reachable", False, f"Start ComfyUI first: {e}")
    
    # COMFYUI_ROOT
    comfyui_root = os.getenv("COMFYUI_ROOT", "") if 'os' in dir() else ""
    if comfyui_root:
        root_path = Path(comfyui_root)
        all_ok &= check(f"COMFYUI_ROOT exists", root_path.exists())
        loras = root_path / "models" / "loras"
        if loras.exists():
            n_loras = len(list(loras.glob("*.safetensors")))
            all_ok &= check(f"LoRAs found: {n_loras}", n_loras > 0)
    
    # Frontend
    frontend = ROOT / "frontend" / "index.html"
    all_ok &= check("Frontend (index.html)", frontend.exists())
    
    # Data directory
    data_dir = ROOT / "data"
    all_ok &= check("Data directory", True)  # will be created on startup
    data_dir.mkdir(exist_ok=True)
    
    print()
    if all_ok:
        print("All checks passed! Run: python -m server.main")
    else:
        print("Some checks failed. Fix the issues above and re-run.")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
