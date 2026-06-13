#!/usr/bin/env python3
"""
Direct ComfyUI test - bypass Prompt Forge server
Sends the exact benchmark prompt that previously worked
"""
import json
import urllib.request
import urllib.parse
import time
from pathlib import Path

COMFYUI_URL = "http://192.168.1.143:8188"

# Exact benchmark prompt that generated a real image
PROMPT = "A golden retriever sitting in the middle of an empty concrete testing facility. Flat overcast lighting. Static tripod wide shot. Documented on archival 16mm Ektachrome film, cyan shadow shift, visible luminance film grain, flat scientific observation framing."

# Exact workflow structure from the working benchmark
# (LoraLoader as node 3, connected to both UNET and CLIP)
WORKFLOW = {
    "1": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "ideogram4_fp8_scaled.safetensors",
            "weight_dtype": "fp8_e4m3fn"
        }
    },
    "2": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen3vl_8b_fp8_scaled.safetensors",
            "type": "ideogram4"
        }
    },
    "3": {
        "class_type": "LoraLoader",
        "inputs": {
            "model": ["1", 0],
            "clip": ["2", 0],
            "lora_name": "ektachrome_style_v1.safetensors",
            "strength_model": 0.8,
            "strength_clip": 0.8
        }
    },
    "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["3", 1],
            "text": PROMPT
        }
    },
    "5": {
        "class_type": "ConditioningZeroOut",
        "inputs": {
            "conditioning": ["4", 0]
        }
    },
    "6": {
        "class_type": "EmptyFlux2LatentImage",
        "inputs": {
            "width": 1024,
            "height": 768,
            "batch_size": 1
        }
    },
    "7": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["3", 0],
            "positive": ["4", 0],
            "negative": ["5", 0],
            "latent_image": ["6", 0],
            "seed": 42,
            "steps": 30,
            "cfg": 3.5,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0
        }
    },
    "8": {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": "flux2-vae.safetensors"
        }
    },
    "9": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["7", 0],
            "vae": ["8", 0]
        }
    },
    "10": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["9", 0],
            "filename_prefix": "direct_test"
        }
    }
}

def queue_prompt():
    """Send workflow to ComfyUI"""
    data = json.dumps({"prompt": WORKFLOW}).encode('utf-8')
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())
    except Exception as e:
        print(f"Error queueing prompt: {e}")
        return None

def get_history(prompt_id):
    """Check if generation completed"""
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as response:
            return json.loads(response.read())
    except:
        return {}

def download_image(filename, subfolder="", folder_type="output"):
    """Download generated image"""
    params = urllib.parse.urlencode({
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type
    })
    
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL}/view?{params}") as response:
            return response.read()
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def main():
    print(f"Testing ComfyUI directly with benchmark prompt...")
    print(f"Prompt: {PROMPT[:80]}...")
    print(f"Using LoRA: ektachrome_style_v1.safetensors")
    print()
    
    # Queue the prompt
    result = queue_prompt()
    if not result:
        print("Failed to queue prompt")
        return
    
    prompt_id = result.get("prompt_id")
    print(f"Queued prompt: {prompt_id}")
    
    # Wait for completion
    print("Waiting for generation...")
    for i in range(30):  # 5 minute timeout
        time.sleep(10)
        history = get_history(prompt_id)
        
        if prompt_id in history:
            outputs = history[prompt_id].get("outputs", {})
            print(f"✓ Generation complete at {(i+1)*10}s")
            
            # Find the saved image
            for node_id, output in outputs.items():
                if "images" in output:
                    for img in output["images"]:
                        filename = img["filename"]
                        subfolder = img.get("subfolder", "")
                        
                        print(f"Image: {filename}")
                        
                        # Download it
                        img_data = download_image(filename, subfolder)
                        if img_data:
                            output_path = Path("/Users/speed/prompt-forge/data/outputs") / f"direct_test_{filename}"
                            output_path.write_bytes(img_data)
                            print(f"Saved: {output_path}")
                            print(f"Size: {len(img_data)} bytes")
                            
                            # Quick variance check
                            from PIL import Image
                            import numpy as np
                            import io
                            
                            img = Image.open(io.BytesIO(img_data))
                            arr = np.array(img)
                            std = arr.std()
                            status = "REAL IMAGE" if std > 30 else "GRAY SCREEN"
                            print(f"Variance: std={std:.1f} → {status}")
            return
        
        print(f"  {((i+1)*10)}s...")
    
    print("Timeout - generation took too long")

if __name__ == "__main__":
    main()
