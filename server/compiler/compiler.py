"""Compile structured composition JSON into ComfyUI workflows."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional
from .capability import ComfyUICapabilities
from .templates import load_template


class WorkflowCompiler:
    """Converts ForgePrompt JSON into ComfyUI API-format workflows."""
    
    def __init__(self, capabilities: ComfyUICapabilities, templates_dir: Path):
        self.capabilities = capabilities
        self.templates_dir = templates_dir
    
    def compile(self, prompt: dict, lora_config: Optional[dict] = None) -> dict:
        """Compile prompt to workflow based on available capabilities."""
        strategy = self.capabilities.best_strategy
        
        # Flux is preferred when available (best quality)
        if self.capabilities.has_flux:
            return self._compile_flux(prompt, lora_config)
        elif strategy == "gligen":
            return self._compile_gligen(prompt, lora_config)
        elif strategy == "mega_prompt":
            return self._compile_mega_prompt(prompt, lora_config)
        else:
            return self._compile_flux(prompt, lora_config)
    
    def _compile_flux(self, prompt: dict, lora_config: Optional[dict]) -> dict:
        """Build a Flux Dev workflow with optional LoRA injection."""
        if lora_config and lora_config.get("lora_name"):
            workflow = load_template("flux_lora", self.templates_dir)
            # Update LoRA node
            if "100" in workflow:
                workflow["100"]["inputs"]["lora_name"] = lora_config["lora_name"]
                workflow["100"]["inputs"]["strength_model"] = lora_config.get("strength", 0.8)
                workflow["100"]["inputs"]["strength_clip"] = lora_config.get("clip_strength", 0.6)
        else:
            workflow = load_template("flux_dev", self.templates_dir)
        
        # Build prompt text from structured data
        prompt_text = self._build_prompt_text(prompt, lora_config)
        
        # Inject into CLIPTextEncodeFlux
        if "4" in workflow:
            workflow["4"]["inputs"]["clip_l"] = prompt_text
            workflow["4"]["inputs"]["t5xxl"] = prompt_text
        
        # Randomize seed
        if "6" in workflow:
            workflow["6"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
        
        return workflow
    
    def _compile_mega_prompt(self, prompt: dict, lora_config: Optional[dict]) -> dict:
        """Build a structured mega-prompt workflow (SDXL fallback)."""
        prompt_text = self._build_prompt_text(prompt, lora_config)
        workflow = load_template("mega_prompt", self.templates_dir)
        
        if "6" in workflow:
            workflow["6"]["inputs"]["text"] = prompt_text
        
        if lora_config and lora_config.get("lora_name"):
            lora_node = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["4", 0],
                    "clip": ["4", 1],
                    "lora_name": lora_config["lora_name"],
                    "strength_model": lora_config.get("strength", 0.8),
                    "strength_clip": lora_config.get("clip_strength", 0.6),
                }
            }
            workflow["100"] = lora_node
            # Rewire KSampler to use LoRA outputs
            if "3" in workflow:
                workflow["3"]["inputs"]["model"] = ["100", 0]
                workflow["3"]["inputs"]["positive"] = ["100", 1]
        
        return workflow
    
    def _compile_gligen(self, prompt: dict, lora_config: Optional[dict]) -> dict:
        """Build GLIGEN workflow with bbox conditioning (SD1.5)."""
        workflow = load_template("gligen_sdxl", self.templates_dir)
        return workflow
    
    def _build_prompt_text(self, prompt: dict, lora_config: Optional[dict]) -> str:
        """Build a natural language prompt from structured JSON data."""
        parts = []
        
        # LoRA trigger words first
        if lora_config:
            triggers = lora_config.get("trigger_words", [])
            if triggers:
                parts.extend(triggers)
        
        # Style description
        style = prompt.get("style_description", {})
        if style:
            style_parts = [str(v) for v in style.values() if v]
            if style_parts:
                parts.extend(style_parts)
        
        # Caption / main subject
        caption = prompt.get("caption", "")
        if caption:
            parts.append(caption)
        
        # Background
        bg = prompt.get("composition", {}).get("background", "")
        if bg and bg != caption:
            parts.append(bg)
        
        # Elements
        for elem in prompt.get("composition", {}).get("elements", []):
            desc = elem.get("desc", elem.get("description", ""))
            if desc and desc not in parts:
                parts.append(desc)
        
        # Negative prompt handling (stored separately)
        # Flux doesn't use negative prompts the same way, but we include them in the text
        
        result = ", ".join(parts) if parts else "a beautiful photograph, high quality"
        return result
