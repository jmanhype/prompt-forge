"""
Design patterns for Prompt Forge.

Key patterns used:
- Strategy: Enrichment approaches (film, digital, illustration, etc.)
- Factory: Component creation (scorers, enrichers, workflow builders)
- Observer: Score feedback and diagnosis
- Registry: Configurable values and signatures
- Template Method: Workflow templates with parameters
- Null Object: Safe defaults for missing components
"""

from .strategy import EnrichmentStrategy, FilmEnrichment, DigitalEnrichment, NoEnrichment
from .factory import EnrichmentFactory, ScorerFactory, WorkflowFactory
from .observer import ScoreObserver, DiagnosisObserver, HeatmapObserver
from .registry import NodeRegistry, CalibrationRegistry, ThresholdRegistry
from .template import WorkflowTemplate
from .null import NullEnrichment, NullScorer

__all__ = [
    # Strategy
    'EnrichmentStrategy', 'FilmEnrichment', 'DigitalEnrichment', 'NoEnrichment',
    # Factory
    'EnrichmentFactory', 'ScorerFactory', 'WorkflowFactory',
    # Observer
    'ScoreObserver', 'DiagnosisObserver', 'HeatmapObserver',
    # Registry
    'NodeRegistry', 'CalibrationRegistry', 'ThresholdRegistry',
    # Template
    'WorkflowTemplate',
    # Null
    'NullEnrichment', 'NullScorer',
]
