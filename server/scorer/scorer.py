"""Composite scoring engine — real CLIP-based scoring with calibrated normalization."""
from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from PIL import Image
from pathlib import Path
from typing import Optional, List

from ..patterns.registry import CalibrationRegistry


@dataclass
class RegionScore:
    """Score for a single bounding box region."""
    label: str
    bbox: list[float]
    clip_score: float = 0.0
    composite: float = 0.0
    diagnosis: str = ""


@dataclass
class ForgeScore:
    """Composite score from all scoring modules."""
    overall: float = 0.0
    style: float = 0.0
    subject: float = 0.0
    composition: float = 0.0
    regions: list[RegionScore] = field(default_factory=list)
    converged: bool = False
    _diagnosis: list[str] = field(default_factory=list)

    def diagnosis(self) -> list[str]:
        """Return diagnosis messages (called as method by engine)."""
        return self._diagnosis

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "style": self.style,
            "subject": self.subject,
            "composition": self.composition,
            "regions": [
                {
                    "label": r.label,
                    "bbox": r.bbox,
                    "clip_score": r.clip_score,
                    "composite": r.composite,
                    "diagnosis": r.diagnosis,
                }
                for r in self.regions
            ],
            "converged": self.converged,
            "diagnosis": self._diagnosis,
        }


class Scorer:
    """Scores generated images against the target composition using CLIP."""

    def __init__(
        self,
        threshold: float = 0.85,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
        calibration_registry: Optional[CalibrationRegistry] = None
    ):
        self.threshold = threshold
        self.model_name = model_name
        self.pretrained = pretrained
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._device = device
        
        # Use registry for normalization, or create default
        if calibration_registry:
            self.calibration = calibration_registry
        else:
            self.calibration = CalibrationRegistry()

    def _ensure_clip(self):
        if self._clip_model is not None:
            return
        import open_clip
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._clip_model, self._clip_preprocess, _ = (
            open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
        )
        self._clip_model = self._clip_model.to(self._device)
        self._clip_model.eval()
        self._clip_tokenizer = open_clip.get_tokenizer(self.model_name)

    @torch.no_grad()
    def _clip_score(self, image: Image.Image, text: str) -> float:
        """Raw CLIP cosine similarity between image and text."""
        self._ensure_clip()
        img_input = self._clip_preprocess(image).unsqueeze(0).to(self._device)
        text_input = self._clip_tokenizer([text]).to(self._device)

        img_feat = self._clip_model.encode_image(img_input)
        txt_feat = self._clip_model.encode_text(text_input)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        return (img_feat @ txt_feat.T).item()

    def normalize(self, raw: float) -> float:
        """Normalize raw CLIP score using calibration registry.
        
        Uses per-model calibration breakpoints loaded from config file.
        Default calibration for ViT-B-32:
          raw < 0.15  → 0.00-0.15 (bad/wrong match)
          raw 0.15-0.22 → 0.15-0.50 (needs mutation)
          raw 0.22-0.30 → 0.50-0.85 (good/close match)
          raw > 0.30  → 0.85-1.00 (excellent/converge)
        """
        return self.calibration.normalize(raw, self.model_name)

    def score(self, image: Image.Image, prompt_data: dict) -> ForgeScore:
        """Score an image against the target prompt data dict.
        
        prompt_data structure:
            {"caption": "...", "composition": {...}, "style_description": {...}}
        """
        result = ForgeScore()
        caption = prompt_data.get("caption", "")

        # Overall CLIP score (full image vs caption)
        if caption:
            raw_overall = self._clip_score(image, caption)
            result.overall = self.normalize(raw_overall)
        else:
            result.overall = 0.0

        # Style score — compose style keywords into a single string
        style = prompt_data.get("style_description", {})
        if style and isinstance(style, dict):
            style_text = " ".join(str(v) for v in style.values() if v)
            if style_text:
                raw_style = self._clip_score(image, style_text)
                result.style = self.normalize(raw_style)
            else:
                result.style = result.overall
        else:
            result.style = result.overall

        # Subject score — score against element labels
        elements = prompt_data.get("composition", {}).get("elements", [])
        if elements:
            subject_scores = []
            for elem in elements:
                label = elem.get("desc", elem.get("label", elem.get("text", "")))
                if label:
                    raw_subj = self._clip_score(image, label)
                    norm_subj = self.normalize(raw_subj)
                    subject_scores.append(norm_subj)
                    result.regions.append(RegionScore(
                        label=label,
                        bbox=elem.get("bbox", []),
                        clip_score=norm_subj,
                        composite=norm_subj,
                        diagnosis=f"{label}: {norm_subj:.2f}" if norm_subj < 0.3 else "",
                    ))
            result.subject = sum(subject_scores) / len(subject_scores) if subject_scores else result.overall
        else:
            result.subject = result.overall

        result.composition = result.overall

        # Convergence check
        result.converged = result.overall >= self.threshold

        # Diagnosis
        if not result.converged:
            if result.style < 0.3:
                result._diagnosis.append(f"style: doesn't match target (score {result.style:.2f})")
            if result.subject < 0.3:
                result._diagnosis.append(f"subject: elements not matching (score {result.subject:.2f})")
            if result.overall < 0.2:
                result._diagnosis.append(f"overall: very low match ({result.overall:.2f})")
            if not result._diagnosis:
                result._diagnosis.append(f"overall: below threshold ({result.overall:.2f} < {self.threshold})")

        return result
