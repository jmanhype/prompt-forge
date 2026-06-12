"""LoRA detector — scan installed LoRAs, extract trigger words, auto-inject."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LoRA:
    filename: str
    path: str
    trigger_words: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    default_strength: float = 0.8
    default_clip_strength: float = 0.6


class LoRADetector:
    """Scans ComfyUI loras/ folder and matches LoRAs to style requests."""
    
    def __init__(self, loras_dir: Path):
        self.loras_dir = loras_dir
        self._loras: list[LoRA] = []
        self._trigger_map: dict[str, LoRA] = {}  # trigger_word → LoRA
        self._api_loras: list[str] = []  # From ComfyUI /object_info
    
    def scan(self, api_loras: Optional[list[str]] = None) -> list[LoRA]:
        """Scan loras directory and extract metadata.
        
        Args:
            api_loras: List of LoRA filenames from ComfyUI /object_info (preferred)
        """
        self._loras = []
        self._trigger_map = {}
        
        # Use API data if provided, otherwise fall back to local scan
        if api_loras:
            self._api_loras = api_loras
            for lora_name in sorted(api_loras):
                lora = self._parse_lora_from_api(lora_name)
                self._loras.append(lora)
                for trigger in lora.trigger_words:
                    self._trigger_map[trigger.lower()] = lora
        elif self.loras_dir.exists():
            for f in sorted(self.loras_dir.glob("*.safetensors")):
                lora = self._parse_lora(f)
                self._loras.append(lora)
                for trigger in lora.trigger_words:
                    self._trigger_map[trigger.lower()] = lora
        
        return self._loras
    
    def _parse_lora_from_api(self, lora_name: str) -> LoRA:
        """Extract trigger words from LoRA filename (from ComfyUI API)."""
        stem = Path(lora_name).stem
        
        # Extract trigger from filename convention
        # e.g., "ektachrome_style_v1" → trigger "ektachrome"
        triggers = []
        name_part = re.split(r'[_-](?:style|v\d|lora|sdxl|sd15)', stem.lower())[0]
        name_part = name_part.replace("_", " ").replace("-", " ").strip()
        if name_part:
            triggers = [name_part]
        
        # Infer style tags from filename
        tags = []
        style_keywords = {
            "vintage": ["vintage", "retro", "old", "classic"],
            "film": ["film", "ektachrome", "kodachrome", "analog"],
            "anime": ["anime", "manga", "cartoon"],
            "realistic": ["realistic", "photo", "real"],
            "artistic": ["art", "paint", "draw", "sketch"],
            "cinematic": ["cinematic", "movie", "film"],
        }
        for tag, keywords in style_keywords.items():
            if any(kw in stem.lower() for kw in keywords):
                tags.append(tag)
        
        return LoRA(
            filename=lora_name,
            path="",
            trigger_words=triggers,
            style_tags=tags,
        )
    
    def _parse_lora(self, path: Path) -> LoRA:
        """Extract trigger words from filename and safetensors metadata."""
        stem = path.stem
        
        # Try reading metadata from safetensors header
        triggers = []
        tags = []
        try:
            metadata = self._read_safetensors_metadata(path)
            if metadata:
                triggers = metadata.get("ss_tag_frequency", {}).keys() if isinstance(metadata.get("ss_tag_frequency"), dict) else []
                # Extract trigger words from training tags
                raw_triggers = metadata.get("trigger_words", "")
                if isinstance(raw_triggers, str) and raw_triggers:
                    triggers.extend([t.strip() for t in raw_triggers.split(",")])
        except Exception:
            pass
        
        # Fallback: extract trigger from filename convention
        # e.g., "ektachrome_style_v1" → trigger "ektachrome"
        if not triggers:
            # Common patterns: name_style_vN, name_vN, name_lora
            name_part = re.split(r'[_-](?:style|v\d|lora|sdxl|sd15)', stem.lower())[0]
            name_part = name_part.replace("_", " ").replace("-", " ").strip()
            if name_part:
                triggers = [name_part]
        
        # Infer style tags from filename
        style_keywords = {
            "vintage": ["vintage", "retro", "old", "classic"],
            "film": ["film", "ektachrome", "kodachrome", "analog"],
            "anime": ["anime", "manga", "cartoon"],
            "realistic": ["realistic", "photo", "real"],
            "artistic": ["art", "paint", "draw", "sketch"],
            "cinematic": ["cinematic", "movie", "film"],
        }
        for tag, keywords in style_keywords.items():
            if any(kw in stem.lower() for kw in keywords):
                tags.append(tag)
        
        return LoRA(
            filename=path.name,
            path=str(path),
            trigger_words=triggers,
            style_tags=tags,
        )
    
    def _read_safetensors_metadata(self, path: Path) -> Optional[dict]:
        """Read metadata from safetensors file header."""
        try:
            with open(path, "rb") as f:
                # First 8 bytes = header size (little-endian uint64)
                header_size = int.from_bytes(f.read(8), "little")
                if header_size > 10_000_000:  # sanity check
                    return None
                header_bytes = f.read(min(header_size, 100_000))
            
            import json
            header = json.loads(header_bytes)
            
            # Metadata is in __metadata__ key
            metadata = header.get("__metadata__", {})
            return metadata if metadata else None
        except Exception:
            return None
    
    def match(self, prompt: dict) -> Optional[dict]:
        """Match prompt style to an installed LoRA. Returns lora config or None."""
        if not self._loras:
            return None
        
        # Check style_description for trigger matches
        style = prompt.get("style_description", {})
        search_text = " ".join(str(v) for v in style.values()).lower()
        
        # Also check caption/description
        caption = prompt.get("caption", "").lower()
        search_text += " " + caption
        
        for trigger, lora in self._trigger_map.items():
            if trigger in search_text:
                return {
                    "lora_name": lora.filename,
                    "trigger_words": lora.trigger_words,
                    "strength": lora.default_strength,
                    "clip_strength": lora.default_clip_strength,
                }
        
        # Check style tags
        for lora in self._loras:
            for tag in lora.style_tags:
                if tag in search_text:
                    return {
                        "lora_name": lora.filename,
                        "trigger_words": lora.trigger_words,
                        "strength": lora.default_strength,
                        "clip_strength": lora.default_clip_strength,
                    }
        
        return None
    
    def list_loras(self) -> list[dict]:
        """Return list of detected LoRAs with their metadata."""
        return [
            {
                "filename": lora.filename,
                "trigger_words": lora.trigger_words,
                "style_tags": lora.style_tags,
                "default_strength": lora.default_strength,
            }
            for lora in self._loras
        ]
