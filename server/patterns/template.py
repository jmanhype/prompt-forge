"""
Template Method Pattern for workflow compilation.

Problems solved:
- Hardcoded workflow template values (#17)
- Hardcoded caption template (#2)
- Hardcoded caption prefix (#5)
- Hardcoded LoRA suffix (#4)

Solution: Define the skeleton of workflow compilation,
parameterize the variable parts, let configuration fill them in.

Trade-offs:
+ Clear structure: Fixed skeleton, variable parts
+ Consistent: All workflows follow the same pattern
+ Configurable: Override any part without rewriting
- Inflexible for radical departures from the template
- Can lead to "template bloat" with too many parameters
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import json


class WorkflowTemplate:
    """Template for ComfyUI workflow compilation.
    
    Defines the structure; parameters fill in the specifics.
    
    Skeleton:
    1. Load model components (UNET, CLIP, VAE, LoRA)
    2. Encode prompt (positive + negative conditioning)
    3. Set up sampling (noise schedule, scheduler, guider)
    4. Generate (latent → sample → decode → save)
    
    Each step has configurable parameters.
    """
    
    # Default parameters - can be overridden per-request
    DEFAULTS = {
        # Model loading
        'unet_name': 'ideogram4_fp8_scaled.safetensors',
        'unet_weight_dtype': 'fp8_e4m3fn',
        'unconditional_model': 'ideogram4_unconditional-Q2_K.gguf',
        'clip_name': 'qwen3vl_8b_fp8_scaled.safetensors',
        'clip_type': 'ideogram4',
        'vae_name': 'flux2-vae.safetensors',
        
        # LoRA
        'lora_enabled': False,
        'lora_name': '',
        'lora_strength_model': 0.8,
        'lora_strength_clip': 0.8,
        
        # Sampling
        'width': 768,
        'height': 768,
        'steps': 28,
        'cfg': 7.0,
        'sampler_name': 'euler',
        'scheduler': 'simple',
        'denoise': 1.0,
        
        # Noise schedule
        'aura_flow_shift': 5.0,
        'cfg_override': 3.0,
        'cfg_start_percent': 0.9,
        'cfg_end_percent': 1.0,
    }
    
    def __init__(self, template_path: Optional[Path] = None, overrides: Optional[Dict] = None):
        """Initialize with optional template file and overrides.
        
        Args:
            template_path: Path to base workflow JSON template
            overrides: Dict of parameter overrides
        """
        self.params = dict(self.DEFAULTS)
        
        # Load base template if provided
        if template_path and template_path.exists():
            with open(template_path) as f:
                self.base_template = json.load(f)
        else:
            self.base_template = None
        
        # Apply overrides
        if overrides:
            self.params.update(overrides)
    
    def compile(self, prompt_text: str, seed: Optional[int] = None) -> dict:
        """Compile a complete ComfyUI workflow.
        
        This is the template method - defines the skeleton.
        Each step is parameterized.
        
        Args:
            prompt_text: The encoded prompt (JSON or plain text)
            seed: Random seed (auto-generated if None)
            
        Returns:
            Complete ComfyUI workflow dict
        """
        import random
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        workflow = {}
        
        # Step 1: Load model components
        workflow.update(self._build_model_loaders())
        
        # Step 2: Apply LoRA if enabled
        if self.params['lora_enabled'] and self.params['lora_name']:
            workflow.update(self._build_lora_loader())
        
        # Step 3: Encode prompt
        workflow.update(self._build_prompt_encoding(prompt_text))
        
        # Step 4: Set up noise schedule
        workflow.update(self._build_noise_schedule())
        
        # Step 5: Set up guider
        workflow.update(self._build_guider())
        
        # Step 6: Generate
        workflow.update(self._build_sampling(seed))
        
        # Step 7: Decode and save
        workflow.update(self._build_decode_and_save())
        
        return workflow
    
    def _build_model_loaders(self) -> dict:
        """Build model loading nodes."""
        nodes = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": self.params['unet_name'],
                    "weight_dtype": self.params['unet_weight_dtype']
                }
            },
            "2": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {
                    "unet_name": self.params['unconditional_model']
                }
            },
            "3": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": self.params['clip_name'],
                    "type": self.params['clip_type']
                }
            }
        }
        return nodes
    
    def _build_lora_loader(self) -> dict:
        """Build LoRA loading node."""
        return {
            "4": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["1", 0],
                    "clip": ["3", 0],
                    "lora_name": self.params['lora_name'],
                    "strength_model": self.params['lora_strength_model'],
                    "strength_clip": self.params['lora_strength_clip']
                }
            }
        }
    
    def _build_prompt_encoding(self, prompt_text: str) -> dict:
        """Build prompt encoding nodes."""
        clip_source = ["4", 1] if self.params['lora_enabled'] else ["3", 0]
        
        return {
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": clip_source,
                    "text": prompt_text
                }
            },
            "6": {
                "class_type": "ConditioningZeroOut",
                "inputs": {
                    "conditioning": ["5", 0]
                }
            }
        }
    
    def _build_noise_schedule(self) -> dict:
        """Build noise schedule nodes."""
        model_source = ["4", 0] if self.params['lora_enabled'] else ["1", 0]
        
        return {
            "7": {
                "class_type": "ModelSamplingAuraFlow",
                "inputs": {
                    "model": model_source,
                    "shift": self.params['aura_flow_shift']
                }
            },
            "8": {
                "class_type": "CFGOverride",
                "inputs": {
                    "model": ["7", 0],
                    "cfg": self.params['cfg_override'],
                    "start_percent": self.params['cfg_start_percent'],
                    "end_percent": self.params['cfg_end_percent']
                }
            }
        }
    
    def _build_guider(self) -> dict:
        """Build dual model guider."""
        return {
            "9": {
                "class_type": "DualModelGuider",
                "inputs": {
                    "model": ["8", 0],
                    "positive": ["5", 0],
                    "cfg": self.params['cfg'],
                    "model_negative": ["2", 0],
                    "negative": ["6", 0]
                }
            }
        }
    
    def _build_sampling(self, seed: int) -> dict:
        """Build sampling nodes."""
        return {
            "10": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": self.params['width'],
                    "height": self.params['height'],
                    "batch_size": 1
                }
            },
            "11": {
                "class_type": "RandomNoise",
                "inputs": {
                    "noise_seed": seed
                }
            },
            "12": {
                "class_type": "KSamplerSelect",
                "inputs": {
                    "sampler_name": self.params['sampler_name']
                }
            },
            "13": {
                "class_type": "BasicScheduler",
                "inputs": {
                    "model": ["8", 0],
                    "scheduler": self.params['scheduler'],
                    "steps": self.params['steps'],
                    "denoise": self.params['denoise']
                }
            },
            "14": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["11", 0],
                    "guider": ["9", 0],
                    "sampler": ["12", 0],
                    "sigmas": ["13", 0],
                    "latent_image": ["10", 0]
                }
            }
        }
    
    def _build_decode_and_save(self) -> dict:
        """Build decode and save nodes."""
        return {
            "15": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": self.params['vae_name']
                }
            },
            "16": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["14", 0],
                    "vae": ["15", 0]
                }
            },
            "17": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "forge_ideo4",
                    "images": ["16", 0]
                }
            }
        }


class PromptTemplate:
    """Template for prompt construction.
    
    Solves: Hardcoded caption template, prefix, and LoRA suffix.
    
    Instead of forcing a template, provides a structured builder
    that respects the user's input.
    """
    
    def __init__(self):
        self.subject: str = ""
        self.setting: str = ""
        self.medium: str = ""
        self.technical: List[str] = []
        self.lora_triggers: List[str] = []
    
    def with_subject(self, text: str) -> 'PromptTemplate':
        """Set the main subject (user's input)."""
        self.subject = text
        return self
    
    def with_setting(self, text: str) -> 'PromptTemplate':
        """Set the environment/setting."""
        self.setting = text
        return self
    
    def with_medium(self, text: str) -> 'PromptTemplate':
        """Set the artistic medium (only if user specified one)."""
        self.medium = text
        return self
    
    def with_technical(self, terms: List[str]) -> 'PromptTemplate':
        """Add technical terms."""
        self.technical.extend(terms)
        return self
    
    def with_lora_triggers(self, triggers: List[str]) -> 'PromptTemplate':
        """Add LoRA trigger words."""
        self.lora_triggers.extend(triggers)
        return self
    
    def build(self) -> str:
        """Build the final prompt text.
        
        Key principle: User's subject is NEVER rewritten.
        Only append context that the user hasn't specified.
        """
        parts = []
        
        # Subject is always first and untouched
        if self.subject:
            parts.append(self.subject)
        
        # Setting only if provided and not already in subject
        if self.setting and self.setting not in self.subject:
            parts.append(self.setting)
        
        # Medium only if user specified it
        if self.medium:
            parts.append(self.medium)
        
        # Technical terms
        if self.technical:
            parts.append(", ".join(self.technical))
        
        # LoRA triggers last
        if self.lora_triggers:
            parts.append(", ".join(self.lora_triggers))
        
        return ". ".join(parts) if parts else ""
    
    def build_json(self) -> dict:
        """Build structured JSON prompt for Ideogram 4.
        
        User's text goes through as-is into the JSON structure.
        No rewriting of the subject.
        """
        caption = self.build()
        
        return {
            "high_level_description": caption or "A detailed image",
            "compositional_decomposition": {
                "background": self.setting or "Background and environment",
                "elements": [
                    {
                        "type": "subject",
                        "bbox": [200, 150, 800, 850],
                        "description": self.subject or "Main subject"
                    }
                ]
            }
        }
