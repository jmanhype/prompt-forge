"""Simple forge engine — text → system prompt → JSON → ComfyUI → image."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx

from .config import config


SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"


class SimpleForge:
    """Generates images using the proven system prompt + Ektachrome LoRA workflow."""

    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT_PATH.read_text().strip()
        self.llm_url = config.LLM_URL
        self.llm_model = config.LLM_MODEL
        self.llm_api_key = config.LLM_API_KEY
        self.comfyui_url = config.COMFYUI_URL  # http://192.168.1.143:8188
        self.lora_name = "ektachrome_style_v1.safetensors"
        self.lora_strength = 0.8

    async def generate(self, description: str) -> dict:
        """Full pipeline: description → JSON prompt → ComfyUI workflow → image."""
        session_id = str(uuid.uuid4())[:8]
        start = time.time()

        # Step 1: Generate structured JSON via LLM
        json_prompt = await self._generate_json(description)

        # Step 2: Build ComfyUI workflow
        workflow = self._build_workflow(json_prompt)

        # Step 3: Submit to ComfyUI and wait
        prompt_id = await self._queue_prompt(workflow)
        image_path = await self._wait_for_image(prompt_id)

        duration_ms = int((time.time() - start) * 1000)

        return {
            "session_id": session_id,
            "description": description,
            "json_prompt": json_prompt,
            "image": image_path,
            "duration_ms": duration_ms,
        }

    async def _generate_json(self, description: str) -> dict:
        """Call LLM with system prompt to generate Ideogram 4 JSON."""
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.llm_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": description},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

        # Extract JSON (handle possible markdown fences)
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            content = "\n".join(lines)

        return json.loads(content)

    def _build_workflow(self, json_prompt: dict) -> dict:
        """Build ComfyUI workflow with Ektachrome LoRA (proven setup from June 10)."""
        prompt_string = json.dumps(json_prompt, ensure_ascii=False)

        return {
            "prompt": {
                "1": {
                    "class_type": "UNETLoader",
                    "inputs": {
                        "unet_name": "ideogram4_fp8_scaled.safetensors",
                        "weight_dtype": "fp8_e4m3fn",
                    },
                },
                "2": {
                    "class_type": "CLIPLoader",
                    "inputs": {
                        "clip_name": "qwen3vl_8b_fp8_scaled.safetensors",
                        "type": "ideogram4",
                    },
                },
                "3": {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "model": ["1", 0],
                        "clip": ["2", 0],
                        "lora_name": self.lora_name,
                        "strength_model": self.lora_strength,
                        "strength_clip": self.lora_strength,
                    },
                },
                "4": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "clip": ["3", 1],
                        "text": prompt_string,
                    },
                },
                "5": {
                    "class_type": "ConditioningZeroOut",
                    "inputs": {"conditioning": ["4", 0]},
                },
                "6": {
                    "class_type": "ModelSamplingAuraFlow",
                    "inputs": {"model": ["3", 0], "shift": 5.0},
                },
                "7": {
                    "class_type": "BasicGuider",
                    "inputs": {
                        "model": ["6", 0],
                        "conditioning": ["4", 0],
                    },
                },
                "8": {
                    "class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": "euler"},
                },
                "9": {
                    "class_type": "BasicScheduler",
                    "inputs": {
                        "model": ["6", 0],
                        "scheduler": "simple",
                        "steps": 30,
                        "denoise": 1.0,
                    },
                },
                "10": {
                    "class_type": "EmptySD3LatentImage",
                    "inputs": {"width": 1024, "height": 768, "batch_size": 1},
                },
                "11": {
                    "class_type": "SamplerCustomAdvanced",
                    "inputs": {
                        "noise": ["12", 0],
                        "guider": ["7", 0],
                        "sampler": ["8", 0],
                        "sigmas": ["9", 0],
                        "latent_image": ["10", 0],
                    },
                },
                "12": {
                    "class_type": "RandomNoise",
                    "inputs": {"noise_seed": 42},
                },
                "13": {
                    "class_type": "VAELoader",
                    "inputs": {"vae_name": "flux2-vae.safetensors"},
                },
                "14": {
                    "class_type": "VAEDecode",
                    "inputs": {
                        "samples": ["11", 0],
                        "vae": ["13", 0],
                    },
                },
                "15": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "filename_prefix": "forge",
                        "images": ["14", 0],
                    },
                },
            },
            "client_id": "prompt-forge",
        }

    async def _queue_prompt(self, workflow: dict) -> str:
        """Submit workflow to ComfyUI and return prompt_id."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.comfyui_url}/prompt",
                json=workflow,
            )
            resp.raise_for_status()
            return resp.json()["prompt_id"]

    async def _wait_for_image(self, prompt_id: str, timeout: int = 300) -> str:
        """Poll ComfyUI history until image is ready, then download."""
        config.ensure_dirs()
        start = time.time()

        async with httpx.AsyncClient() as client:
            while time.time() - start < timeout:
                resp = await client.get(f"{self.comfyui_url}/history/{prompt_id}")
                history = resp.json()

                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status", {}).get("status_str", "")

                    if status == "success":
                        # Find the output image
                        for node_id, outputs in entry.get("outputs", {}).items():
                            for img in outputs.get("images", []):
                                filename = img["filename"]
                                # Download image
                                img_resp = await client.get(
                                    f"{self.comfyui_url}/view",
                                    params={"filename": filename},
                                )
                                local_path = config.OUTPUTS_DIR / f"{prompt_id[:8]}.png"
                                local_path.write_bytes(img_resp.content)
                                return str(local_path)

                    elif status == "error":
                        msgs = entry["status"].get("messages", [])
                        error_msg = ""
                        for m in msgs:
                            if m[0] == "execution_error":
                                error_msg = m[1].get("exception_message", "")
                        raise RuntimeError(f"ComfyUI error: {error_msg}")

                await asyncio.sleep(3)

        raise TimeoutError(f"Generation timed out after {timeout}s")


# Need asyncio for the wait
import asyncio
