"""Composite scoring engine — real CLIP-based scoring."""
from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from PIL import Image
from typing import Optional


@dataclass
class RegionScore:
    element_id: str
    label: str
    bbox_iou: float = 0.0
    dino_similarity: float = 0.0
    clip_score: float = 0.0
    present: bool = True

    @property
    def composite(self) -> float:
        if not self.present:
            return 0.0
        return 0.4 * self.bbox_iou + 0.4 * self.dino_similarity + 0.2 * self.clip_score


@dataclass
class ForgeScore:
    composition: float = 0.0
    style: float = 0.0
    subject: float = 0.0
    overall: float = 0.0
    regions: list[RegionScore] = field(default_factory=list)
    converged: bool = False
    threshold: float = 0.85

    def diagnosis(self) -> list[str]:
        """Generate human-readable diagnosis of what failed."""
        issues = []
        for region in self.regions:
            if not region.present:
                issues.append(f"{region.label}: element missing from output")
            elif region.bbox_iou < 0.5:
                issues.append(f"{region.label}: misplaced (IoU {region.bbox_iou:.2f})")
            elif region.clip_score < 0.6:
                issues.append(f"{region.label}: doesn't match description (CLIP {region.clip_score:.2f})")

        if self.style < 0.7:
            issues.append(f"style: doesn't match target (CLIP {self.style:.2f})")
        if self.subject < 0.6:
            issues.append(f"subject: not clear in image (CLIP {self.subject:.2f})")

        return issues if issues else ["All elements look good"]

    def to_dict(self) -> dict:
        return {
            "composition": round(self.composition, 3),
            "style": round(self.style, 3),
            "subject": round(self.subject, 3),
            "overall": round(self.overall, 3),
            "converged": self.converged,
            "threshold": self.threshold,
            "regions": [
                {
                    "id": r.element_id,
                    "label": r.label,
                    "bbox_iou": round(r.bbox_iou, 3),
                    "clip_score": round(r.clip_score, 3),
                    "present": r.present,
                    "composite": round(r.composite, 3),
                }
                for r in self.regions
            ],
            "diagnosis": self.diagnosis(),
        }


class Scorer:
    """Evaluate generated output against target using CLIP similarity."""

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load_clip(self):
        """Lazy-load CLIP model."""
        if self._model is not None:
            return

        import open_clip

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self._model.eval()

    def _clip_score(self, image: Image.Image, text: str) -> float:
        """Compute CLIP similarity between image and text. Returns 0-1."""
        self._load_clip()

        image_input = self._preprocess(image).unsqueeze(0)
        text_input = self._tokenizer([text])

        with torch.no_grad():
            image_features = self._model.encode_image(image_input)
            text_features = self._model.encode_text(text_input)

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Cosine similarity (range roughly -1 to 1, typically 0.15-0.35 for random)
            similarity = (image_features @ text_features.T).item()

        # Map to 0-1 range based on ViT-B-32 typical similarity distribution:
        # Random pairs: ~0.13, weak: 0.16-0.19, moderate: 0.19-0.24, strong: 0.24-0.30
        normalized = max(0.0, min(1.0, (similarity - 0.13) / 0.17))
        return normalized

    def score(
        self,
        generated_image: Image.Image,
        target_prompt: dict,
    ) -> ForgeScore:
        """Score a generated image against the target prompt using CLIP."""
        regions = []

        # Build text descriptions for scoring
        caption = target_prompt.get("caption", "")
        style_desc = target_prompt.get("style_description", {})
        style_text = " ".join(str(v) for v in style_desc.values()) if style_desc else ""
        full_text = f"{caption} {style_text}".strip()

        # Score overall style (CLIP similarity to full prompt)
        if full_text:
            style_score = self._clip_score(generated_image, full_text)
        else:
            style_score = 0.5

        # Score subject (CLIP similarity to caption only)
        if caption:
            subject_score = self._clip_score(generated_image, caption)
        else:
            subject_score = style_score

        # Score per-element regions
        elements = target_prompt.get("composition", {}).get("elements", [])
        for elem in elements:
            label = elem.get("desc", elem.get("label", "unknown"))
            elem_id = elem.get("id", "unknown")

            # CLIP score for this element
            elem_score = self._clip_score(generated_image, label) if label else 0.5

            region = RegionScore(
                element_id=elem_id,
                label=label[:50],
                bbox_iou=elem_score,  # Use CLIP as proxy for placement
                dino_similarity=elem_score,
                clip_score=elem_score,
                present=elem_score > 0.3,
            )
            regions.append(region)

        # Composition = average of element scores
        composition = (
            sum(r.bbox_iou for r in regions) / max(len(regions), 1)
            if regions
            else style_score
        )

        overall = 0.4 * composition + 0.3 * style_score + 0.3 * subject_score

        score = ForgeScore(
            composition=composition,
            style=style_score,
            subject=subject_score,
            overall=overall,
            regions=regions,
            converged=overall >= self.threshold,
            threshold=self.threshold,
        )

        return score
