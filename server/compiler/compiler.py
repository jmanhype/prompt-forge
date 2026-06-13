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
        """Compile prompt to workflow. Ideogram 4 preferred when LoRA trained on it."""
        if lora_config and self.capabilities.has_ideogram4:
            return self._compile_ideogram4(prompt, lora_config)
        strategy = self.capabilities.best_strategy
        if strategy == "ideogram4":
            return self._compile_ideogram4(prompt, lora_config)
        elif strategy == "flux":
            return self._compile_flux(prompt, lora_config)
        else:
            return self._compile_mega_prompt(prompt, lora_config)

    def _compile_ideogram4(self, prompt: dict, lora_config: Optional[dict]) -> dict:
        """Build Ideogram 4 workflow — verified working pipeline.

        Architecture (from actual training runs):
        - UNETLoader: ideogram4_fp8_scaled.safetensors (fp8_e4m3fn)
        - CLIPLoader: qwen3vl_8b_fp8_scaled.safetensors (type=ideogram4)
        - LoraLoader: injects LoRA into model+clip
        - CLIPTextEncode: positive conditioning
        - ConditioningZeroOut: negative (zero of positive, not text negative)
        - EmptyFlux2LatentImage: latent space (Flux2-compatible)
        - KSampler: 30 steps, cfg 3.5, euler, simple scheduler
        - VAELoader: flux2-vae.safetensors
        - VAEDecode -> SaveImage
        """
        if lora_config and lora_config.get("lora_name"):
            workflow = load_template("ideogram4_lora", self.templates_dir)
            if "3" in workflow:
                workflow["3"]["inputs"]["lora_name"] = lora_config["lora_name"]
                workflow["3"]["inputs"]["strength_model"] = lora_config.get("strength", 0.8)
                workflow["3"]["inputs"]["strength_clip"] = lora_config.get("clip_strength", 0.8)
        else:
            workflow = load_template("ideogram4_base", self.templates_dir)

        # Build prompt text
        prompt_text = self._build_prompt_text(prompt, lora_config)

        # Inject into CLIPTextEncode (node 4)
        if "4" in workflow:
            workflow["4"]["inputs"]["text"] = prompt_text

        # Randomize seed
        if "7" in workflow:
            workflow["7"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)

        return workflow

    def _compile_flux(self, prompt: dict, lora_config: Optional[dict]) -> dict:
        """Build a Flux Dev workflow."""
        if lora_config and lora_config.get("lora_name"):
            workflow = load_template("flux_lora", self.templates_dir)
            if "100" in workflow:
                workflow["100"]["inputs"]["lora_name"] = lora_config["lora_name"]
                workflow["100"]["inputs"]["strength_model"] = lora_config.get("strength", 0.8)
                workflow["100"]["inputs"]["strength_clip"] = lora_config.get("clip_strength", 0.6)
        else:
            workflow = load_template("flux_dev", self.templates_dir)

        prompt_text = self._build_prompt_text(prompt, lora_config)
        if "4" in workflow:
            workflow["4"]["inputs"]["clip_l"] = prompt_text
            workflow["4"]["inputs"]["t5xxl"] = prompt_text
        if "6" in workflow:
            workflow["6"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
        return workflow

    def _compile_mega_prompt(self, prompt: dict, lora_config: Optional[dict]) -> dict:
        """Build a structured mega-prompt workflow (SDXL fallback)."""
        prompt_text = self._build_prompt_text(prompt, lora_config)
        workflow = load_template("mega_prompt", self.templates_dir)
        if "6" in workflow:
            workflow["6"]["inputs"]["text"] = prompt_text
        return workflow

    def _build_prompt_text(self, prompt: dict, lora_config: Optional[dict]) -> str:
        """Build plain text prompt for Ideogram 4.
        
        Based on verified working benchmark prompts — plain sentences work best:
        "A golden retriever sitting in the middle of an empty concrete testing 
        facility. Flat overcast lighting. Static tripod wide shot. Documented on 
        archival 16mm Ektachrome film, cyan shadow shift, visible luminance film 
        grain, flat scientific observation framing."
        
        Key formula: subject + location + lighting + camera + medium/film
        """
        import sys
        
        # Extract caption from either format (legacy or cocktailpeanut)
        caption = prompt.get("caption") or prompt.get("high_level_description", "")
        
        # Extract elements from either format
        elements = []
        comp = prompt.get("composition") or prompt.get("compositional_decomposition", {})
        if isinstance(comp, dict):
            elements = comp.get("elements", [])
        
        print(f"\n[COMPILER] caption ({len(caption)} chars): '{caption[:80]}...'", file=sys.stderr)
        
        # If caption is already rich (200+ chars, has sentences), use it directly
        if len(caption) > 100 and "." in caption:
            result = caption
        else:
            # Build a rich prompt from parts
            parts = []
            
            # LoRA trigger words first
            if lora_config:
                triggers = lora_config.get("trigger_words", [])
                if triggers:
                    parts.append(", ".join(triggers))
            
            # Subject (the caption)
            if caption:
                parts.append(caption if len(caption) > 20 else f"A photograph of {caption}")
            
            # Element descriptions
            for elem in elements:
                desc = elem.get("desc", elem.get("description", ""))
                if desc and desc not in caption:
                    parts.append(desc)
            
            result = ". ".join(parts) if parts else "a detailed photograph"
        
        # Add LoRA film characteristics if applicable
        if lora_config and "ektachrome" in str(lora_config).lower():
            if "Ektachrome" not in result and "ektachrome" not in result:
                result += ". Documented on archival 16mm Ektachrome film, cyan shadow shift, visible luminance film grain, flat scientific observation framing."
        
        print(f"[COMPILER] final prompt ({len(result)} chars): '{result[:100]}...'", file=sys.stderr)
        return result
