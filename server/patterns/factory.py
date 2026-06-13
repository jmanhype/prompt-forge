"""
Factory Pattern for component creation.

Problems solved:
- Hardcoded CLIP model selection (#8)
- Hardcoded LoRA detection logic (#4)
- Hardcoded workflow compilation (#17)

Solution: Factory classes that create components based on:
- Configuration
- Runtime context
- User preferences

Trade-offs:
+ Centralized creation logic
+ Easy to add new variants
+ Configuration-driven
- Additional abstraction layer
- More code to maintain
"""

from typing import Optional, Dict, Any
from .strategy import (
    EnrichmentStrategy, NoEnrichment, FilmEnrichment, 
    DigitalEnrichment, IllustrationEnrichment, ConceptArtEnrichment
)


class EnrichmentFactory:
    """Creates enrichment strategies based on context.
    
    Priority order:
    1. User-specified strategy (from config/request)
    2. Strategy that matches prompt content
    3. Strategy that matches LoRA
    4. NoEnrichment (Null Object)
    """
    
    STRATEGIES = {
        'none': NoEnrichment,
        'film': FilmEnrichment,
        'digital': DigitalEnrichment,
        'illustration': IllustrationEnrichment,
        'concept_art': ConceptArtEnrichment,
    }
    
    @classmethod
    def create(
        cls,
        prompt: dict,
        lora_config: Optional[dict] = None,
        preferred: Optional[str] = None
    ) -> EnrichmentStrategy:
        """Create appropriate enrichment strategy.
        
        Args:
            prompt: Current prompt data
            lora_config: LoRA configuration if loaded
            preferred: User-preferred strategy name
            
        Returns:
            EnrichmentStrategy instance
        """
        # Priority 1: User preference
        if preferred and preferred in cls.STRATEGIES:
            return cls.STRATEGIES[preferred]()
        
        # Priority 2: Auto-detect from prompt content
        for strategy_cls in [
            ConceptArtEnrichment,  # Check first (more specific)
            IllustrationEnrichment,
            FilmEnrichment,
            DigitalEnrichment,
        ]:
            strategy = strategy_cls()
            if strategy.should_apply(prompt, lora_config):
                return strategy
        
        # Priority 3: Default to no enrichment
        return NoEnrichment()
    
    @classmethod
    def list_available(cls) -> list[str]:
        """List all available strategy names."""
        return list(cls.STRATEGIES.keys())


class ScorerFactory:
    """Creates scorers with configurable models.
    
    Solves: Hardcoded ViT-B-32 model selection.
    """
    
    @staticmethod
    def create_scorer(
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        threshold: float = 0.85,
        device: Optional[str] = None
    ):
        """Create a scorer with specified model.
        
        Args:
            model_name: CLIP model variant
            pretrained: Pretrained checkpoint
            threshold: Convergence threshold
            device: torch device (auto-detected if None)
            
        Returns:
            Configured Scorer instance
        """
        from ..scorer.scorer import Scorer
        return Scorer(
            model_name=model_name,
            pretrained=pretrained,
            threshold=threshold,
            device=device
        )
    
    @staticmethod
    def create_calibrated_scorer(
        model_name: str,
        calibration_data: dict,
        threshold: float = 0.85
    ):
        """Create scorer with custom calibration breakpoints.
        
        Args:
            model_name: CLIP model name
            calibration_data: Breakpoints from calibration study
            threshold: Convergence threshold
            
        Returns:
            Scorer with custom normalization
        """
        from ..scorer.scorer import Scorer
        return Scorer(
            model_name=model_name,
            calibration_breakpoints=calibration_data.get('breakpoints'),
            threshold=threshold
        )


class WorkflowFactory:
    """Creates workflow compilers with auto-detected capabilities.
    
    Solves: Hardcoded workflow template values and model names.
    """
    
    @staticmethod
    async def create_compiler(comfyui_url: str, templates_dir: str):
        """Create compiler with auto-detected ComfyUI capabilities.
        
        Args:
            comfyui_url: ComfyUI server URL
            templates_dir: Path to workflow templates
            
        Returns:
            Configured WorkflowCompiler
        """
        from ..compiler.capability import probe_capabilities
        from ..compiler.compiler import WorkflowCompiler
        from pathlib import Path
        
        caps = await probe_capabilities(comfyui_url)
        return WorkflowCompiler(caps, Path(templates_dir))
    
    @staticmethod
    def create_with_params(
        capabilities,
        templates_dir: str,
        default_resolution: tuple[int, int] = (768, 768),
        default_steps: int = 28,
        default_sampler: str = "euler",
        default_scheduler: str = "simple"
    ):
        """Create compiler with explicit parameters.
        
        Args:
            capabilities: ComfyUI capabilities
            templates_dir: Path to templates
            default_resolution: Default (width, height)
            default_steps: Default sampling steps
            default_sampler: Default sampler name
            default_scheduler: Default scheduler name
            
        Returns:
            Configured WorkflowCompiler
        """
        from ..compiler.compiler import WorkflowCompiler
        from pathlib import Path
        
        compiler = WorkflowCompiler(capabilities, Path(templates_dir))
        compiler.default_resolution = default_resolution
        compiler.default_steps = default_steps
        compiler.default_sampler = default_sampler
        compiler.default_scheduler = default_scheduler
        return compiler
