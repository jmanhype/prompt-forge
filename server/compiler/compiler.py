"""Compile structured composition JSON into ComfyUI workflows."""
from __future__ import annotations

import json
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
        
        if strategy == "mega_prompt":
            return self._compile_mega_prompt(prompt, lora_config)
        elif strategy == "gligen":
            return self._compile_gligen(prompt, lora_config)
        elif strategy == "attention_couple":
            return self._compile_attention_couple(prompt, lora_config)
        else:
            # Fallback
            return self._compile_mega_prompt(prompt, lora_config)
    
    def _compile_mega_prompt(self, prompt: dict, lora_config: Optional[dict]) -> dict:
        """Build a structured mega-prompt workflow (works with any checkpoint)."""
        # Build the prompt string from structured data
        parts = []
        
        # Style prefix
        style = prompt.get("style_description", {})
        if style:
            style_parts = [v for v in style.values() if v]
            parts.append(f"({', '.join(style_parts)}:1.2)")
        
        # Background
        bg = prompt.get("composition", {}).get("background", "")
        if bg:
            parts.append(bg)
        
        # Elements with weights
        for elem in prompt.get("composition", {}).get("elements", []):
            desc = elem.get("desc", elem.get("description", ""))
            if desc:
                parts.append(f"({desc}:1.3)")
        
        full_prompt = ", ".join(parts)
        
        # Add LoRA triggers
        if lora_config:
            lora_prompt_parts = [f"{t}" for t in lora_config.get("trigger_words", [])]
            if lora_prompt_parts:
                full_prompt = ", ".join(lora_prompt_parts) + ", " + full_prompt
        
        # Build minimal ComfyUI API workflow
        workflow = load_template("mega_prompt", self.templates_dir)
        
        # Inject prompt text
        if "6" in workflow:  # Standard CLIP text encode node
            workflow["6"]["inputs"]["text"] = full_prompt
        
        # Add LoRA loader nodes if configured
        if lora_config and lora_config.get("lora_name"):
            lora_node = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["4", 0],  # connected to checkpoint
                    "clip": ["4", 1],
                    "lora_name": lora_config["lora_name"],
                    "strength_model": lora_config.get("strength", 0.8),
                    "strength_clip": lora_config.get("clip_strength", 0.6),
                }
            }
            workflow["100"] = lora_node
        
        return workflow
    
    def _compile_gligen(self, prompt: dict, lora_config: Optional[dict]) -> dict:
        """Build GLIGEN workflow with bbox conditioning."""
        workflow = load_template("gligen_sdxl", self.templates_dir)
        # TODO: inject bbox coordinates into GLIGEN nodes
        return workflow
    
    def _compile_attention_couple(self, prompt: dict, lora_config: Optional[dict]) -> dict:
        """Build Attention Couple workflow for regional control."""
        workflow = load_template("attention_couple", self.templates_dir)
        # TODO: inject regional prompts and masks
        return workflow
