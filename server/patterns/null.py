"""
Null Object Pattern for safe defaults.

Problems solved:
- Missing enrichment strategies
- Missing scorers
- Missing workflow compilers
- Avoiding null checks throughout the codebase

Solution: Provide "do nothing" implementations that are safe to call.

Trade-offs:
+ No null checks needed
+ Safe defaults
+ Simplifies client code
- Can hide errors if null objects are used unexpectedly
- Need to be careful about side effects (or lack thereof)
"""

from typing import List, Optional, Dict, Any
from .strategy import EnrichmentStrategy


class NullEnrichment(EnrichmentStrategy):
    """Null Object for enrichment - does nothing.
    
    Use when: No enrichment should be applied.
    Safe to call all methods - they return input unchanged.
    """
    
    def get_name(self) -> str:
        return "null"
    
    def enrich_caption(self, caption: str) -> str:
        """Return caption unchanged."""
        return caption
    
    def enrich_elements(self, elements: List[dict]) -> List[dict]:
        """Return elements unchanged."""
        return elements
    
    def get_environment_phrases(self) -> List[str]:
        """Return empty list."""
        return []
    
    def should_apply(self, prompt: dict, lora_config: Optional[dict] = None) -> bool:
        """Always returns True (null object is always applicable)."""
        return True


class NullScorer:
    """Null Object for scorer - returns neutral scores.
    
    Use when: Scoring is disabled or unavailable.
    Returns scores that indicate "needs more information".
    """
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
    
    def score(self, image: Any, prompt_data: dict) -> Any:
        """Return a neutral ForgeScore."""
        from ..scorer.scorer import ForgeScore
        
        score = ForgeScore()
        score.overall = 0.5  # Neutral - neither good nor bad
        score.style = 0.5
        score.subject = 0.5
        score.composition = 0.5
        score.converged = False
        score._diagnosis = ["Scoring disabled - using null scorer"]
        return score
    
    def normalize(self, raw: float) -> float:
        """Identity normalization."""
        return raw


class NullMutator:
    """Null Object for mutator - does nothing.
    
    Use when: Mutation is disabled or all scores are good.
    Returns prompt unchanged.
    """
    
    def mutate(self, prompt: dict, score: Any) -> tuple[dict, List[str]]:
        """Return prompt unchanged with empty changes list."""
        return prompt, ["No mutations applied (null mutator)"]
    
    def should_use_llm(self, score: Any, iteration: int, prev_score: Any = None) -> bool:
        """Always returns False."""
        return False


class NullConnector:
    """Null Object for ComfyUI connector - simulates generation.
    
    Use when: ComfyUI is not available (testing, demo mode).
    Returns mock results.
    """
    
    def __init__(self, url: str = ""):
        self.url = url
    
    async def is_reachable(self) -> bool:
        """Always returns False."""
        return False
    
    async def queue_prompt(self, workflow: dict) -> str:
        """Return a mock prompt ID."""
        return "null-prompt-id"
    
    async def wait_for_completion(self, prompt_id: str, timeout: int = 300):
        """Yield immediate completion."""
        from ..connector.comfyui import ProgressUpdate
        yield ProgressUpdate(type="executed", value=1.0)
    
    async def get_outputs(self, prompt_id: str, output_dir: Any) -> List[str]:
        """Return empty list."""
        return []
    
    async def generate(self, workflow: dict, output_dir: Any, timeout: int = 300):
        """Yield mock generation result."""
        from ..connector.comfyui import ProgressUpdate, GenerationResult
        
        yield ProgressUpdate(type="queued", message="Mock generation")
        yield ProgressUpdate(type="executed", value=1.0)
        yield GenerationResult(
            prompt_id="null-prompt-id",
            images=[],
            duration_ms=0,
            error="Null connector - no actual generation"
        )


class NullAnalyzer:
    """Null Object for image analyzer - returns minimal analysis.
    
    Use when: Florence-2 or other analyzers are unavailable.
    Returns empty/minimal analysis results.
    """
    
    def analyze(self, image: Any) -> Any:
        """Return minimal analysis result."""
        from ..analyzer.florence import AnalysisResult
        
        return AnalysisResult(
            caption="An image",
            objects=[],
            regions=[]
        )
    
    def to_ideogram_json(self, analysis: Any) -> dict:
        """Convert to minimal Ideogram JSON format."""
        return {
            "caption": "An image",
            "composition": {
                "background": "Background",
                "elements": []
            },
            "style_description": {},
            "negative_prompt": []
        }


class NullLoRADetector:
    """Null Object for LoRA detector - finds no LoRAs.
    
    Use when: LoRA detection is disabled or no LoRAs are available.
    """
    
    def __init__(self, loras_dir: Any = None):
        self.loras_dir = loras_dir
    
    def scan(self, api_loras: Optional[List[str]] = None) -> None:
        """Do nothing."""
        pass
    
    def match(self, prompt: dict) -> Optional[dict]:
        """Always returns None (no LoRA matched)."""
        return None
    
    def list_available(self) -> List[str]:
        """Return empty list."""
        return []
