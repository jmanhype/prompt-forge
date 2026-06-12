"""Workflow template registry — load and customize ComfyUI workflow JSONs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


# Cache loaded templates
_template_cache: dict[str, dict] = {}


def load_template(name: str, templates_dir: Path) -> dict:
    """Load a workflow template by name. Returns a deep copy."""
    if name not in _template_cache:
        path = templates_dir / f"{name}.json"
        if not path.exists():
            # Return minimal fallback workflow
            return _fallback_workflow()
        
        with open(path) as f:
            _template_cache[name] = json.load(f)
    
    # Return a copy so mutations don't affect cache
    return json.loads(json.dumps(_template_cache[name]))


def list_templates(templates_dir: Path) -> list[str]:
    """List available template names."""
    return [f.stem for f in templates_dir.glob("*.json")]


def _fallback_workflow() -> dict:
    """Minimal ComfyUI workflow: checkpoint → CLIP encode → KSampler → VAE decode → save."""
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "sd_xl_base_1.0.safetensors"
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1,
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "a beautiful landscape",
                "clip": ["4", 1],
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "ugly, blurry, low quality, watermark",
                "clip": ["4", 1],
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2],
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "forge",
                "images": ["8", 0],
            }
        }
    }
