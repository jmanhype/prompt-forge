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
        - UnetLoaderGGUF: ideogram4_unconditional-Q2_K.gguf
        - CLIPLoader: qwen3vl_8b_fp8_scaled.safetensors (type=ideogram4)
        - LoraLoader: injects LoRA into model+clip
        - CLIPTextEncode: JSON prompt with bounding boxes
        - ConditioningZeroOut: negative (zero of positive, not text negative)
        - ModelSamplingAuraFlow: shift=5.0 noise schedule
        - CFGOverride: cfg=3.0, start=0.9, end=1.0
        - DualModelGuider: dual model guidance with unconditional model
        - SamplerCustomAdvanced: 28 steps, euler, simple scheduler
        - VAELoader: flux2-vae.safetensors
        - VAEDecode -> SaveImage
        """
        if lora_config and lora_config.get("lora_name"):
            workflow = load_template("ideogram4_lora", self.templates_dir)
            if "4" in workflow:
                workflow["4"]["inputs"]["lora_name"] = lora_config["lora_name"]
                workflow["4"]["inputs"]["strength_model"] = lora_config.get("strength", 0.8)
                workflow["4"]["inputs"]["strength_clip"] = lora_config.get("clip_strength", 0.8)
        else:
            workflow = load_template("ideogram4_lora", self.templates_dir)
            # Remove LoRA node if no LoRA
            if "4" in workflow:
                # Reconnect CLIP and model to skip LoRA
                if "5" in workflow:
                    workflow["5"]["inputs"]["clip"] = ["3", 0]
                if "7" in workflow:
                    workflow["7"]["inputs"]["model"] = ["1", 0]

        # Build JSON prompt text
        prompt_text = self._build_prompt_text(prompt, lora_config)

        # Inject into CLIPTextEncode (node 5)
        if "5" in workflow:
            workflow["5"]["inputs"]["text"] = prompt_text

        # Randomize seed
        if "11" in workflow:
            workflow["11"]["inputs"]["noise_seed"] = random.randint(0, 2**32 - 1)

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
        """Build JSON prompt with bounding boxes for Ideogram 4.
        
        JSON prompts bypass the safety filter by changing how the encoder
        tokenizes the input. Bounding boxes provide spatial layout control.
        
        Schema:
        {
          "high_level_description": "...",
          "compositional_decomposition": {
            "background": "...",
            "elements": [
              {"type": "subject", "bbox": [y_min, x_min, y_max, x_max], "description": "..."}
            ]
          }
        }
        
        Bbox format: [y_min, x_min, y_max, x_max] in 0-1000 scale.
        """
        import sys
        
        # Extract caption and composition
        caption = prompt.get("caption", "")
        composition = prompt.get("composition", {})
        background = composition.get("background", "")
        elements = composition.get("elements", [])
        
        print(f"\n[COMPILER] Building JSON prompt, caption ({len(caption)} chars): '{caption[:80]}...'", file=sys.stderr)
        
        # Build structured JSON prompt
        json_prompt = {
            "high_level_description": caption or "A detailed photograph",
            "compositional_decomposition": {
                "background": background or "Background and environment surrounding the subject.",
                "elements": []
            }
        }
        
        # Add elements with bounding boxes (0-1000 scale, [y_min, x_min, y_max, x_max])
        for i, elem in enumerate(elements):
            desc = elem.get("desc", elem.get("description", ""))
            if not desc:
                continue
            
            # Use bbox if provided, otherwise create a default centered box
            bbox = elem.get("bbox")
            if not bbox or len(bbox) != 4:
                # Default: center the element, size based on index
                # First element: large center box
                # Subsequent elements: smaller boxes around it
                if i == 0:
                    bbox = [200, 150, 800, 850]  # Main subject: large center
                else:
                    # Place smaller elements around
                    positions = [
                        [100, 100, 400, 500],   # Top-left
                        [100, 500, 400, 900],   # Top-right
                        [600, 100, 900, 500],   # Bottom-left
                        [600, 500, 900, 900],   # Bottom-right
                    ]
                    bbox = positions[(i - 1) % len(positions)]
            
            json_prompt["compositional_decomposition"]["elements"].append({
                "type": elem.get("type", "object"),
                "bbox": bbox,
                "description": desc
            })
        
        # If no elements were added, add a default subject element
        if not json_prompt["compositional_decomposition"]["elements"]:
            json_prompt["compositional_decomposition"]["elements"].append({
                "type": "subject",
                "bbox": [200, 150, 800, 850],
                "description": caption or "Main subject"
            })
        
        # Add LoRA style hints if applicable
        if lora_config:
            triggers = lora_config.get("trigger_words", [])
            if triggers and "ektachrome" in " ".join(triggers).lower():
                # Prepend Ektachrome style to the description
                json_prompt["high_level_description"] = (
                    "Ektachrome film photography, vintage 1960s Kodak Ektachrome film, "
                    "faded colors, visible grain, " + caption
                )
        
        # Convert to JSON string
        result = json.dumps(json_prompt, indent=2)
        print(f"[COMPILER] JSON prompt ({len(result)} chars): '{result[:100]}...'", file=sys.stderr)
        
        return result
