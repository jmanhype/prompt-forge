"""
Strategy Pattern for prompt enrichment.

Problem solved: Hardcoded FILM_DETAILS, CAMERA_DETAILS, SETTING_DETAILS, SUBJECT_ENRICHMENTS
that force Ektachrome aesthetics on everything.

Solution: Define enrichment strategies that can be swapped based on:
- User's stated aesthetic
- Loaded LoRA
- Prompt content analysis
- Configuration

Trade-offs:
+ Flexible: Add new aesthetics without touching core code
+ Contextual: Choose strategy based on prompt content
+ Testable: Each strategy can be tested independently
- More complex: Interface + implementations vs single list
- Requires context: Need to determine which strategy to use
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class EnrichmentStrategy(ABC):
    """Base interface for all enrichment strategies."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Human-readable name of this strategy."""
        pass
    
    @abstractmethod
    def enrich_caption(self, caption: str) -> str:
        """Add style/medium details to caption."""
        pass
    
    @abstractmethod
    def enrich_elements(self, elements: List[dict]) -> List[dict]:
        """Add detail to element descriptions."""
        pass
    
    @abstractmethod
    def get_environment_phrases(self) -> List[str]:
        """Return environment-specific phrases."""
        pass
    
    def should_apply(self, prompt: dict, lora_config: Optional[dict] = None) -> bool:
        """Check if this strategy should be applied to the given prompt."""
        return True


class NoEnrichment(EnrichmentStrategy):
    """Null Object pattern: No enrichment applied."""
    
    def get_name(self) -> str:
        return "none"
    
    def enrich_caption(self, caption: str) -> str:
        return caption
    
    def enrich_elements(self, elements: List[dict]) -> List[dict]:
        return elements
    
    def get_environment_phrases(self) -> List[str]:
        return []


class FilmEnrichment(EnrichmentStrategy):
    """Film/archival aesthetic enrichment.
    
    Use when: User mentions film, archival, documentary, or loads film LoRA.
    """
    
    FILM_DETAILS = [
        "visible luminance film grain throughout the frame",
        "cyan color shift in the shadow areas",
        "slight optical gate weave at the frame edges",
        "faded archival quality with subtle color desaturation",
        "natural lens vignetting at corners",
    ]
    
    CAMERA_DETAILS = [
        "shot from a static tripod",
        "medium format camera perspective",
        "slightly underexposed for rich shadow detail",
    ]
    
    def get_name(self) -> str:
        return "film"
    
    def enrich_caption(self, caption: str) -> str:
        if "film" not in caption.lower() and "archival" not in caption.lower():
            return f"{caption}, archival documentary photograph"
        return caption
    
    def enrich_elements(self, elements: List[dict]) -> List[dict]:
        # Add film-specific details to elements
        for elem in elements:
            desc = elem.get("description", "")
            if "film" not in desc.lower():
                elem["description"] = f"{desc}, captured on film"
        return elements
    
    def get_environment_phrases(self) -> List[str]:
        return [
            "documented for archival purposes",
            "muted earth tones in surroundings",
            "natural ambient lighting",
        ]
    
    def should_apply(self, prompt: dict, lora_config: Optional[dict] = None) -> bool:
        """Apply if prompt mentions film/archival or LoRA is film-based."""
        caption = prompt.get("caption", "").lower()
        if any(kw in caption for kw in ["film", "archival", "documentary", "vintage"]):
            return True
        if lora_config and "ektachrome" in str(lora_config).lower():
            return True
        return False


class DigitalEnrichment(EnrichmentStrategy):
    """Digital photography aesthetic."""
    
    def get_name(self) -> str:
        return "digital"
    
    def enrich_caption(self, caption: str) -> str:
        if "digital" not in caption.lower() and "photograph" not in caption.lower():
            return f"{caption}, high-resolution digital photograph"
        return caption
    
    def enrich_elements(self, elements: List[dict]) -> List[dict]:
        for elem in elements:
            desc = elem.get("description", "")
            if "sharp" not in desc.lower() and "detail" not in desc.lower():
                elem["description"] = f"{desc}, sharp detail"
        return elements
    
    def get_environment_phrases(self) -> List[str]:
        return [
            "clean modern aesthetic",
            "balanced exposure",
            "natural colors",
        ]
    
    def should_apply(self, prompt: dict, lora_config: Optional[dict] = None) -> bool:
        caption = prompt.get("caption", "").lower()
        return any(kw in caption for kw in ["photo", "digital", "modern", "sharp"])


class IllustrationEnrichment(EnrichmentStrategy):
    """Illustration/concept art aesthetic."""
    
    def get_name(self) -> str:
        return "illustration"
    
    def enrich_caption(self, caption: str) -> str:
        if "illustration" not in caption.lower() and "art" not in caption.lower():
            return f"{caption}, detailed illustration"
        return caption
    
    def enrich_elements(self, elements: List[dict]) -> List[dict]:
        for elem in elements:
            desc = elem.get("description", "")
            if "detail" not in desc.lower():
                elem["description"] = f"{desc}, highly detailed"
        return elements
    
    def get_environment_phrases(self) -> List[str]:
        return [
            "dynamic composition",
            "dramatic lighting",
            "stylized rendering",
        ]
    
    def should_apply(self, prompt: dict, lora_config: Optional[dict] = None) -> bool:
        caption = prompt.get("caption", "").lower()
        return any(kw in caption for kw in ["illustration", "concept", "art", "painting", "drawing"])


class ConceptArtEnrichment(EnrichmentStrategy):
    """Concept art for fantasy/sci-fi subjects."""
    
    def get_name(self) -> str:
        return "concept_art"
    
    def enrich_caption(self, caption: str) -> str:
        if "concept" not in caption.lower():
            return f"{caption}, professional concept art"
        return caption
    
    def enrich_elements(self, elements: List[dict]) -> List[dict]:
        for elem in elements:
            desc = elem.get("description", "")
            if "detail" not in desc.lower():
                elem["description"] = f"{desc}, intricate detail"
        return elements
    
    def get_environment_phrases(self) -> List[str]:
        return [
            "cinematic composition",
            "dramatic lighting",
            "professional concept art quality",
        ]
    
    def should_apply(self, prompt: dict, lora_config: Optional[dict] = None) -> bool:
        caption = prompt.get("caption", "").lower()
        return any(kw in caption for kw in ["fantasy", "sci-fi", "dragon", "robot", "creature", "alien"])
