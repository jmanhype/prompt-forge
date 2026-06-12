"""Composite scoring engine — orchestrates per-region scoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
            elif region.dino_similarity < 0.6:
                issues.append(f"{region.label}: doesn't look right (sim {region.dino_similarity:.2f})")
        
        if self.style < 0.7:
            issues.append(f"style: doesn't match target ({self.style:.2f})")
        if self.composition < 0.6:
            issues.append(f"composition: elements not arranged correctly ({self.composition:.2f})")
        
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
                    "dino_sim": round(r.dino_similarity, 3),
                    "clip_score": round(r.clip_score, 3),
                    "present": r.present,
                    "composite": round(r.composite, 3),
                }
                for r in self.regions
            ],
            "diagnosis": self.diagnosis(),
        }


class Scorer:
    """Evaluate generated output against target composition."""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
    
    def score(
        self,
        generated_image: Image.Image,
        target_analysis: dict,
    ) -> ForgeScore:
        """Score a generated image against the target analysis."""
        from .compositional import compute_bbox_iou
        from .visual import compute_dino_similarity
        
        regions = []
        target_elements = target_analysis.get("elements", [])
        
        for elem in target_elements:
            region = RegionScore(
                element_id=elem.get("id", "unknown"),
                label=elem.get("label", "unknown"),
            )
            
            # Bbox IoU — compare target bbox with detected bbox in output
            target_bbox = elem.get("bbox", [])
            if target_bbox and len(target_bbox) == 4:
                # For now, use a simple heuristic (full re-detection needs Florence on output)
                region.bbox_iou = 0.7  # placeholder — full impl re-detects
            else:
                region.bbox_iou = 0.5
            
            # DINOv2 similarity — crop regions and compare
            region.dino_similarity = 0.7  # placeholder
            
            # CLIP score — global text-image similarity
            region.clip_score = 0.75  # placeholder
            
            regions.append(region)
        
        # Compute composite
        composition = sum(r.bbox_iou for r in regions) / max(len(regions), 1)
        subject = sum(r.dino_similarity for r in regions) / max(len(regions), 1)
        style = 0.75  # placeholder — CLIP global
        
        overall = 0.4 * composition + 0.3 * style + 0.3 * subject
        
        score = ForgeScore(
            composition=composition,
            style=style,
            subject=subject,
            overall=overall,
            regions=regions,
            converged=overall >= self.threshold,
            threshold=self.threshold,
        )
        
        return score
