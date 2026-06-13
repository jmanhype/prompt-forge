"""
Registry Pattern for configurable values.

Problems solved:
- Hardcoded NODE_SIGNATURES in capability.py (#6)
- Hardcoded normalization breakpoints (#9)
- Magic numbers scattered throughout
- Hardcoded LoRA parameters (#17)

Solution: Load configuration from files, provide defaults, allow runtime updates.

Trade-offs:
+ Configurable without code changes
+ Version-controlled configuration
+ Runtime updates possible
- Configuration loading overhead
- Need to handle missing/invalid configs gracefully
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json


class NodeRegistry:
    """Registry for ComfyUI node signatures.
    
    Solves: Hardcoded NODE_SIGNATURES dict (#6).
    Load from config file, allow runtime updates.
    """
    
    DEFAULT_SIGNATURES = {
        "has_gligen": ["GLIGENLoader", "GLIGENTextBoxApply"],
        "has_attention_couple": ["AttentionCouple", "AttentionCoupleBase"],
        "has_ipadapter_region": ["IPAdapterRegionalConditioning", "IPAdapterRegional"],
        "has_flux_guidance": ["FluxGuidance", "FluxDisableGuidance"],
        "has_regional_condition": ["RegionalConditioning", "RegionalPrompter", "RegionalSampler"],
        "has_flux": ["CLIPTextEncodeFlux", "ModelSamplingFlux"],
        "has_ideogram4": ["IdeogramV4", "Ideogram4Scheduler", "CLIPLoader"],
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with optional config file.
        
        Args:
            config_path: Path to node_signatures.json
        """
        self.signatures = dict(self.DEFAULT_SIGNATURES)
        if config_path and config_path.exists():
            self.load(config_path)
    
    def load(self, path: Path) -> None:
        """Load signatures from JSON file."""
        with open(path) as f:
            loaded = json.load(f)
            self.signatures.update(loaded)
    
    def save(self, path: Path) -> None:
        """Save current signatures to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.signatures, f, indent=2)
    
    def get(self, capability: str) -> List[str]:
        """Get node signatures for a capability."""
        return self.signatures.get(capability, [])
    
    def register(self, capability: str, signatures: List[str]) -> None:
        """Register new capability signatures."""
        self.signatures[capability] = signatures
    
    def check_capability(self, capability: str, available_nodes: set) -> bool:
        """Check if a capability is available."""
        signatures = self.get(capability)
        return any(sig in available_nodes for sig in signatures)


class CalibrationRegistry:
    """Registry for CLIP score calibration breakpoints.
    
    Solves: Hardcoded normalization breakpoints (#9).
    Store per-model calibration data.
    """
    
    # Default calibration for ViT-B-32 with Ideogram 4
    DEFAULT_BREAKPOINTS = {
        "ViT-B-32": {
            "breakpoints": [
                (0.15, 0.0, 0.15),   # raw < 0.15 → 0.00-0.15
                (0.22, 0.15, 0.35),  # raw 0.15-0.22 → 0.15-0.50
                (0.30, 0.50, 0.35),  # raw 0.22-0.30 → 0.50-0.85
                (0.35, 0.85, 0.15),  # raw > 0.30 → 0.85-1.00
            ],
            "description": "Calibrated on 65 measurements, 5 images, 13 prompts"
        }
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with optional config file.
        
        Args:
            config_path: Path to calibration_data.json
        """
        self.calibrations = dict(self.DEFAULT_BREAKPOINTS)
        if config_path and config_path.exists():
            self.load(config_path)
    
    def load(self, path: Path) -> None:
        """Load calibrations from JSON file."""
        with open(path) as f:
            loaded = json.load(f)
            # Convert breakpoint tuples back from lists
            for model, data in loaded.items():
                if 'breakpoints' in data:
                    data['breakpoints'] = [tuple(bp) for bp in data['breakpoints']]
            self.calibrations.update(loaded)
    
    def save(self, path: Path) -> None:
        """Save current calibrations to JSON file."""
        # Convert tuples to lists for JSON serialization
        serializable = {}
        for model, data in self.calibrations.items():
            serializable[model] = dict(data)
            if 'breakpoints' in data:
                serializable[model]['breakpoints'] = [list(bp) for bp in data['breakpoints']]
        
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    def get_breakpoints(self, model_name: str) -> List[tuple]:
        """Get calibration breakpoints for a model.
        
        Returns:
            List of (raw_threshold, normalized_start, normalized_range) tuples
        """
        if model_name in self.calibrations:
            return self.calibrations[model_name]['breakpoints']
        # Fallback to default
        return self.DEFAULT_BREAKPOINTS["ViT-B-32"]["breakpoints"]
    
    def normalize(self, raw_score: float, model_name: str) -> float:
        """Normalize raw CLIP score using calibrated breakpoints.
        
        Args:
            raw_score: Raw cosine similarity
            model_name: CLIP model name
            
        Returns:
            Normalized score in [0, 1]
        """
        breakpoints = self.get_breakpoints(model_name)
        
        # Find which segment this score falls into
        prev_threshold = 0.0
        for threshold, norm_start, norm_range in breakpoints:
            if raw_score < threshold:
                # Linear interpolation within this segment
                segment_size = threshold - prev_threshold
                if segment_size == 0:
                    return norm_start
                position = (raw_score - prev_threshold) / segment_size
                return norm_start + position * norm_range
            prev_threshold = threshold
        
        # Beyond last threshold
        last_threshold, last_start, last_range = breakpoints[-1]
        return min(1.0, last_start + last_range)
    
    def register_calibration(self, model_name: str, breakpoints: List[tuple], description: str = "") -> None:
        """Register new model calibration."""
        self.calibrations[model_name] = {
            'breakpoints': breakpoints,
            'description': description
        }


class ThresholdRegistry:
    """Registry for scoring and mutation thresholds.
    
    Solves: Magic numbers for thresholds (#10, #15).
    Derive thresholds from convergence threshold.
    """
    
    def __init__(self, convergence_threshold: float = 0.85):
        """Initialize with base convergence threshold.
        
        Args:
            convergence_threshold: Target score for convergence
        """
        self.convergence_threshold = convergence_threshold
        self._ratios = {
            'style_diagnosis': 0.6,
            'subject_diagnosis': 0.6,
            'overall_diagnosis': 0.4,
            'mutation_trigger': 0.7,
            'plateau_threshold': 0.03,
        }
    
    def get(self, threshold_name: str) -> float:
        """Get a derived threshold.
        
        Args:
            threshold_name: Name of threshold
            
        Returns:
            Calculated threshold value
        """
        ratio = self._ratios.get(threshold_name, 1.0)
        if threshold_name == 'plateau_threshold':
            # Absolute value, not derived
            return ratio
        return self.convergence_threshold * ratio
    
    def set_ratio(self, threshold_name: str, ratio: float) -> None:
        """Set a threshold ratio."""
        self._ratios[threshold_name] = ratio
    
    def get_all(self) -> Dict[str, float]:
        """Get all calculated thresholds."""
        return {name: self.get(name) for name in self._ratios.keys()}


class ModelRegistry:
    """Registry for model configurations.
    
    Solves: Hardcoded model names and paths (#8, #17).
    """
    
    DEFAULT_MODELS = {
        "clip": {
            "ViT-B-32": {
                "pretrained": "laion2b_s34b_b79k",
                "description": "Fast, well-calibrated"
            },
            "ViT-L-14": {
                "pretrained": "laion2b_s32b_b82k",
                "description": "Higher quality, slower"
            }
        },
        "unet": {
            "ideogram4_fp8_scaled.safetensors": {
                "weight_dtype": "fp8_e4m3fn",
                "architecture": "ideogram4"
            }
        },
        "clip_loader": {
            "qwen3vl_8b_fp8_scaled.safetensors": {
                "type": "ideogram4"
            }
        },
        "vae": {
            "flux2-vae.safetensors": {
                "architecture": "flux2"
            }
        }
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with optional config file."""
        self.models = dict(self.DEFAULT_MODELS)
        if config_path and config_path.exists():
            self.load(config_path)
    
    def load(self, path: Path) -> None:
        """Load model configs from JSON."""
        with open(path) as f:
            loaded = json.load(f)
            # Deep merge
            for category, models in loaded.items():
                if category not in self.models:
                    self.models[category] = {}
                self.models[category].update(models)
    
    def get(self, category: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model configuration."""
        return self.models.get(category, {}).get(model_name)
    
    def list_models(self, category: str) -> List[str]:
        """List available models in a category."""
        return list(self.models.get(category, {}).keys())
    
    def register(self, category: str, model_name: str, config: Dict[str, Any]) -> None:
        """Register a new model."""
        if category not in self.models:
            self.models[category] = {}
        self.models[category][model_name] = config
