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
        """Build Ideogram 4 native JSON prompt from structured data.
        
        Ideogram 4's Qwen3VL CLIP loader parses structured JSON with:
        - high_level_description: 1-2 sentence overview
        - style_description: aesthetics, lighting, medium, color palette
        - compositional_deconstruction: background + elements with bboxes
        
        This gives much better composition control and avoids safety filter
        false-positives from ambiguous plain text.
        """
        import json as _json
        
        # Build style_description
        style = prompt.get("style_description", {})
        if not style:
            # Infer from prompt data or use sensible defaults
            style = {
                "aesthetics": "professional photography, sharp detail, natural tones",
                "lighting": "soft ambient lighting",
                "medium": "photograph",
                "color_palette": ["#FFFFFF", "#333333", "#666666", "#999999"],
            }
        
        # Ensure color_palette exists and is valid
        if "color_palette" not in style:
            style["color_palette"] = ["#FFFFFF", "#333333", "#666666"]
        
        # Build high_level_description
        caption = prompt.get("caption", "")
        if not caption:
            # Build from elements
            elements = prompt.get("composition", {}).get("elements", [])
            subjects = [e.get("desc", e.get("description", "")) for e in elements[:3] if e.get("desc") or e.get("description")]
            caption = ", ".join(subjects) if subjects else "a photograph"
        
        # Add LoRA trigger words to description
        trigger_prefix = ""
        if lora_config:
            triggers = lora_config.get("trigger_words", [])
            if triggers:
                trigger_prefix = ", ".join(triggers) + ", "
        
        high_level = trigger_prefix + caption if trigger_prefix else caption
        
        # Build compositional_deconstruction
        bg = prompt.get("composition", {}).get("background", "")
        if not bg:
            bg = caption  # fallback
        
        # Build elements with proper Ideogram 4 format
        ideo_elements = []
        raw_elements = prompt.get("composition", {}).get("elements", [])
        
        for i, elem in enumerate(raw_elements):
            elem_type = elem.get("type", "obj")
            desc = elem.get("desc", elem.get("description", elem.get("label", "")))
            
            if not desc:
                continue
            
            entry = {"type": elem_type}
            
            # Add bbox (convert normalized 0-1 to Ideogram's 0-1000 coordinate system)
            bbox = elem.get("bbox")
            if bbox and len(bbox) == 4:
                # Input is [x1, y1, x2, y2] normalized 0-1
                # Ideogram expects [y_min, x_min, y_max, x_max] in 0-1000
                x1, y1, x2, y2 = bbox
                entry["bbox"] = [
                    round(y1 * 1000),
                    round(x1 * 1000),
                    round(y2 * 1000),
                    round(x2 * 1000),
                ]
            
            if elem_type == "text":
                entry["text"] = elem.get("text", desc)
                entry["desc"] = f"Typography: {desc}"
            else:
                entry["desc"] = desc
            
            # Color palette per element
            colors = elem.get("color_palette")
            if colors:
                entry["color_palette"] = colors
            
            ideo_elements.append(entry)
        
        # If no elements, create a basic one from the caption
        if not ideo_elements:
            ideo_elements.append({
                "type": "obj",
                "bbox": [200, 200, 800, 800],  # centered
                "desc": caption,
            })
        
        # Assemble the full Ideogram 4 JSON prompt
        ideo_prompt = {
            "high_level_description": high_level,
            "style_description": style,
            "compositional_deconstruction": {
                "background": bg + " No text, no watermark, no logo, no clutter.",
                "elements": ideo_elements,
            }
        }
        
        return _json.dumps(ideo_prompt)
