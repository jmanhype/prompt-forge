"""
Observer Pattern for score feedback and diagnosis.

Problems solved:
- Hardcoded diagnosis thresholds (#10)
- Hardcoded heatmap colors (#18)
- Tightly coupled score→feedback logic

Solution: Observers subscribe to score changes and react:
- DiagnosisObserver: Generates diagnosis messages
- HeatmapObserver: Generates heatmap data (frontend concern)
- LoggingObserver: Logs score changes

Trade-offs:
+ Decoupled: Scoring doesn't know about diagnosis/heatmap
+ Extensible: Add new observers without changing scorer
+ Testable: Each observer can be tested independently
- Event ordering: Observers run in registration order
- Debugging: Harder to trace through event chain
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json


class ScoreObserver(ABC):
    """Base interface for score observers."""
    
    @abstractmethod
    def on_score_update(self, score: Any, iteration: int) -> None:
        """Called when a new score is computed.
        
        Args:
            score: ForgeScore instance
            iteration: Current iteration number
        """
        pass


class DiagnosisObserver(ScoreObserver):
    """Generates diagnosis messages based on configurable thresholds.
    
    Solves: Hardcoded 0.3/0.2 threshold magic numbers.
    """
    
    def __init__(
        self,
        convergence_threshold: float = 0.85,
        style_threshold_ratio: float = 0.6,
        subject_threshold_ratio: float = 0.6,
        overall_threshold_ratio: float = 0.4
    ):
        """Initialize with configurable threshold ratios.
        
        Args:
            convergence_threshold: Target convergence score
            style_threshold_ratio: Ratio of convergence threshold for style diagnosis
            subject_threshold_ratio: Ratio of convergence threshold for subject diagnosis
            overall_threshold_ratio: Ratio of convergence threshold for overall diagnosis
        """
        self.convergence_threshold = convergence_threshold
        self.style_threshold = convergence_threshold * style_threshold_ratio
        self.subject_threshold = convergence_threshold * subject_threshold_ratio
        self.overall_threshold = convergence_threshold * overall_threshold_ratio
        self.messages: List[str] = []
    
    def on_score_update(self, score: Any, iteration: int) -> None:
        """Generate diagnosis messages for current score."""
        self.messages = []
        
        if not score.converged:
            if score.style < self.style_threshold:
                self.messages.append(
                    f"style: below threshold (score {score.style:.2f} < {self.style_threshold:.2f})"
                )
            if score.subject < self.subject_threshold:
                self.messages.append(
                    f"subject: elements not matching (score {score.subject:.2f} < {self.subject_threshold:.2f})"
                )
            if score.overall < self.overall_threshold:
                self.messages.append(
                    f"overall: very low match ({score.overall:.2f} < {self.overall_threshold:.2f})"
                )
            
            # Check individual regions
            for region in score.regions:
                if region.composite < self.subject_threshold:
                    self.messages.append(
                        f"region '{region.label}': low match ({region.composite:.2f})"
                    )
            
            if not self.messages:
                self.messages.append(
                    f"overall: below threshold ({score.overall:.2f} < {self.convergence_threshold})"
                )
    
    def get_messages(self) -> List[str]:
        """Get current diagnosis messages."""
        return self.messages


class HeatmapObserver(ScoreObserver):
    """Generates heatmap data for frontend visualization.
    
    Solves: Hardcoded heatmap colors in backend (#18).
    Backend sends scores, frontend applies colors.
    """
    
    def __init__(self):
        self.heatmap_data: Dict[str, Any] = {
            'regions': [],
            'overall_score': 0.0,
            'converged': False
        }
    
    def on_score_update(self, score: Any, iteration: int) -> None:
        """Generate heatmap data from scores.
        
        Note: Colors are NOT applied here - that's a frontend concern.
        We send raw scores and let CSS/JS handle visualization.
        """
        regions = []
        for i, region in enumerate(score.regions):
            regions.append({
                'id': str(i),
                'label': region.label,
                'bbox': region.bbox,
                'score': round(region.composite, 3),
                'diagnosis': region.diagnosis or f"Score: {region.composite:.2f}"
            })
        
        self.heatmap_data = {
            'regions': regions,
            'overall_score': round(score.overall, 3),
            'converged': score.converged
        }
    
    def get_heatmap_data(self) -> Dict[str, Any]:
        """Get current heatmap data."""
        return self.heatmap_data


class LoggingObserver(ScoreObserver):
    """Logs score changes for debugging."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.history: List[Dict[str, Any]] = []
    
    def on_score_update(self, score: Any, iteration: int) -> None:
        """Log score to history and optionally to file."""
        entry = {
            'iteration': iteration,
            'overall': score.overall,
            'style': score.style,
            'subject': score.subject,
            'composition': score.composition,
            'converged': score.converged,
            'num_regions': len(score.regions)
        }
        self.history.append(entry)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get score history."""
        return self.history


class MutationTriggerObserver(ScoreObserver):
    """Triggers mutations based on score changes.
    
    Solves: Hardcoded mutation thresholds (#15).
    """
    
    def __init__(
        self,
        convergence_threshold: float = 0.85,
        plateau_threshold: float = 0.03,
        low_score_ratio: float = 0.7
    ):
        """Initialize with configurable thresholds.
        
        Args:
            convergence_threshold: Target score
            plateau_threshold: Minimum improvement to avoid LLM mutation
            low_score_ratio: Ratio below which mutations are triggered
        """
        self.convergence_threshold = convergence_threshold
        self.plateau_threshold = plateau_threshold
        self.low_score_ratio = low_score_ratio
        self.low_score_threshold = convergence_threshold * low_score_ratio
        self.previous_score: Optional[float] = None
        self.should_mutate = False
        self.should_use_llm = False
    
    def on_score_update(self, score: Any, iteration: int) -> None:
        """Determine if mutations should be triggered."""
        self.should_mutate = not score.converged
        
        # Check if we've plateaued
        self.should_use_llm = False
        if self.previous_score is not None and iteration >= 3:
            improvement = score.overall - self.previous_score
            if improvement < self.plateau_threshold:
                self.should_use_llm = True
        
        self.previous_score = score.overall
    
    def get_mutation_decision(self) -> Dict[str, bool]:
        """Get mutation decision."""
        return {
            'should_mutate': self.should_mutate,
            'should_use_llm': self.should_use_llm
        }


class ObserverManager:
    """Manages multiple observers and notifies them of score updates."""
    
    def __init__(self):
        self.observers: List[ScoreObserver] = []
    
    def register(self, observer: ScoreObserver) -> None:
        """Register an observer."""
        self.observers.append(observer)
    
    def unregister(self, observer: ScoreObserver) -> None:
        """Unregister an observer."""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def notify_all(self, score: Any, iteration: int) -> None:
        """Notify all registered observers."""
        for observer in self.observers:
            observer.on_score_update(score, iteration)
