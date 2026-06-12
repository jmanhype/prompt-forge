#!/usr/bin/env python3
"""Install workflow templates into ComfyUI's workflow folder."""
import shutil
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

ROOT = Path(__file__).parent.parent
TEMPLATES = ROOT / "workflows" / "templates"
COMFYUI_ROOT = Path(os.getenv("COMFYUI_ROOT", ""))


def main():
    if not COMFYUI_ROOT.exists():
        print(f"COMFYUI_ROOT not found: {COMFYUI_ROOT}")
        print("Set COMFYUI_ROOT in .env to your ComfyUI install path")
        return 1
    
    target = COMFYUI_ROOT / "user" / "default" / "workflows" / "prompt-forge"
    target.mkdir(parents=True, exist_ok=True)
    
    templates = list(TEMPLATES.glob("*.json"))
    if not templates:
        print("No workflow templates found in", TEMPLATES)
        return 1
    
    for t in templates:
        shutil.copy2(t, target / t.name)
        print(f"  Installed: {t.name}")
    
    print(f"\n{len(templates)} templates installed to {target}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
